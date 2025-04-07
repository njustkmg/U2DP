import torch
import torch.nn.functional
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import clip
import spacy 

def calculate_sim(nlp, caps_gen, clipfets, simi_limit, simi_limit_weak, num_token, nouns, nouns_feature, model_clip):
    sims_list = []
    ignore_idx = []
    ignore_idx_toclipselect = []
    
    # 归一化 nouns_feature
    nouns_feature = nouns_feature / nouns_feature.norm(dim=1, keepdim=True)
    
    with torch.no_grad():  # 确保整个计算不需要梯度
        for i, (cap, clip_features) in enumerate(zip(caps_gen, clipfets)):
                # 归一化 clip_features
                clip_features = clip_features.unsqueeze(0).float()
                clip_features /= clip_features.norm(dim=1, keepdim=True)
                
                # 对 cap生成文本特征
                text = clip.tokenize([cap]).to("cuda")
                text_features = model_clip.encode_text(text).float()
                text_features /= text_features.norm(dim=1, keepdim=True)
                
                # 计算相似性
                similarity = (clip_features @ text_features[0].T).item()
                sims_list.append(similarity)
                
                # 清理不再使用的变量
                del text, text_features
                torch.cuda.empty_cache()
                
                ### 处理不匹配数据
                if similarity < simi_limit or len(cap.split(' ')) < 5:
                    ignore_idx.append(i)
                    similarity_nouns = (clip_features @ nouns_feature.T)[0]
                    topk_values, topk_indices = torch.topk(similarity_nouns, 2)
                    ignore_idx_toclipselect.append(f"it is a {nouns[topk_indices[0]]}")
                    continue
                
                ### 处理弱匹配数据
                if similarity < simi_limit_weak:
                    # words = cap.split(' ')
                    doc = nlp(cap)
                    words = [chunk.text for chunk in doc.noun_chunks]
                    text = clip.tokenize(words).to("cuda")
                    words_feature = model_clip.encode_text(text).float()
                    
                    ignore_idx.append(i)
                    similarity_nouns = (clip_features @ words_feature.T)[0]
                    # print(len(similarity_nouns))
                    topk = min(len(similarity_nouns), num_token)

                    topk_values, topk_indices = torch.topk(similarity_nouns, topk)

                    num_token = min(num_token, len(topk_indices))
                    if num_token == 1:
                        ignore_idx_toclipselect.append(f"it is a {words[topk_indices[0]]}")
                    elif num_token == 2:
                        ignore_idx_toclipselect.append(f"it is {words[topk_indices[0]]} and {words[topk_indices[1]]}")
                    elif num_token == 3:
                        ignore_idx_toclipselect.append(f"it is {words[topk_indices[0]]} and {words[topk_indices[1]]} and {words[topk_indices[2]]}")
                    elif num_token == 4:
                        ignore_idx_toclipselect.append(f"it is {words[topk_indices[0]]} and {words[topk_indices[1]]} and {words[topk_indices[2]]} and {words[topk_indices[3]]}")
                    elif num_token == 5:
                        ignore_idx_toclipselect.append(f"it is {words[topk_indices[0]]} and {words[topk_indices[1]]} and {words[topk_indices[2]]} and {words[topk_indices[3]]} and {words[topk_indices[4]]}")

                    
                    # 清理弱匹配过程的临时变量
                    del text, words_feature
                    torch.cuda.empty_cache()
            
    
    return sims_list, ignore_idx, ignore_idx_toclipselect

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.ablation = opt.ablation
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.nlp = spacy.load("en_core_web_sm")

    def forward(self, epoch, model_clip,nouns,nouns_feature, clip_feats, vocab, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, unlabels):
        out = {}

        ## 打印信息
        # print(f'fc_feats.shape:{fc_feats.shape}')
        # print(f'att_feats.shape:{att_feats.shape}')
        # print(f'att_masks.shape:{att_masks.shape}')
        # print(f'labels.shape:{labels.shape}')
        # print(f'labels[0]:{labels[0]}')
        # print(f'att_masks[0]:{att_masks[0]}')
        # print(f'unlabels:{unlabels}')
        # print("####################################")
        if not sc_flag and epoch < self.opt.gamma:
            # weight = torch.tensor([0 if wei == 1 else 1 for wei in unlabels], dtype=torch.float32).to("cuda")
            loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        
        ## semi-supervised
        elif not sc_flag and epoch >= self.opt.gamma :

            ## supervised loss
            label_indices = (unlabels == 0).nonzero(as_tuple=False).squeeze(-1)
            loss_s = self.crit(self.model(fc_feats[label_indices], att_feats[label_indices], labels[label_indices], att_masks[label_indices]), labels[:, 1:][label_indices], masks[:,1:][label_indices])

            ## unsupervised loss
            with torch.no_grad():
                unlabel_indices = (unlabels == 1).nonzero(as_tuple=False).squeeze(-1)
                seq =self.model(fc_feats[unlabel_indices], att_feats[unlabel_indices], att_masks[unlabel_indices], mode='sample')[0].data
            
            sents = utils.decode_sequence(vocab, seq)
            # print(f'len(sents):{len(sents)} ')
            # print(sents)
            clip_feats = clip_feats[unlabel_indices]
            sim_list, ignore_idxs, ignore_idx_toclipselect = calculate_sim( self.nlp, sents, clip_feats, self.opt.beta, self.opt.alpha, self.opt.num_token,nouns, nouns_feature, model_clip)
            for ignore_idx, clipselect in zip(ignore_idxs, ignore_idx_toclipselect):
                sents[ignore_idx] = clipselect
            word_map = {v: k for k, v in vocab.items()}
            clipselect_toids = list(map(lambda c: [int(word_map[w]) for w in c.split(' ')], sents)) 

            # print(f'after sents:{sents} ')
            # labels = labels[:,1:]
            # print(f'real caption :{utils.decode_sequence(vocab, labels[unlabel_indices])}')
            # print(f'after labels.shape:{labels.shape}')
            pseudo_labels = torch.tensor([])
            
            for unlabel_idx, clipselect in zip(unlabel_indices, clipselect_toids):
                # print(len(labels[unlabel_idx]))
                # print(len(len(clipselect)))
                pad_length = labels.shape[-1] - len(clipselect)
                temp = torch.nn.functional.pad(torch.tensor(clipselect), (0, pad_length),value = 0)
                if pseudo_labels.numel() == 0:  # 判断是否为空
                    pseudo_labels = temp.unsqueeze(0)  # 添加第一个元素并保持维度
                else:
                    pseudo_labels = torch.cat((pseudo_labels, temp.unsqueeze(0)), dim=0)
            
            # print(f'after change labels.shape:{labels.shape}')
            calculate_loss_indices = range(len(unlabel_indices))
            
            if self.ablation > 0:
                if self.ablation == 1:
                    calculate_loss_indices = [i for i, value in enumerate(sim_list) if value >=  self.opt.alpha]
                else :
                    calculate_loss_indices = [i for i, value in enumerate(sim_list) if value >=  self.opt.beta]
                
            if len(calculate_loss_indices) :
                loss_u = self.crit(self.model(fc_feats[unlabel_indices[calculate_loss_indices]], att_feats[unlabel_indices[calculate_loss_indices]], pseudo_labels[calculate_loss_indices].to("cuda"), att_masks[unlabel_indices[calculate_loss_indices]]), pseudo_labels[calculate_loss_indices].to("cuda"), masks[:,1:][unlabel_indices[calculate_loss_indices]])

                loss = loss_s + self.opt.lambda1 * loss_u
            else :
                loss = loss_s
        
        ## reinforce
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method':'sample'}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
