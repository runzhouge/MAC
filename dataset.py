import numpy as np
from math import sqrt
import os
import random
import pickle
home_dir = os.environ['HOME']
from sklearn.metrics.pairwise import cosine_similarity

#========================================================
random.seed(999)
#========================================================


def calculate_IoU(i0,i1):
    union=(min(i0[0],i1[0]) , max(i0[1],i1[1]))
    inter=(max(i0[0],i1[0]) , min(i0[1],i1[1]))
    iou=1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
'''
def calculate_nIoL(base,sliding_clip):
    inter=(max(base[0],sliding_clip[0]) , min(base[1],sliding_clip[1]))
    inter_l=inter[1]-inter[0]
    length=sliding_clip[1]-sliding_clip[0]
    nIoL=1.0*(length-inter_l)/length
    return nIoL

class TrainingDataSet(object):
    def __init__(self, sliding_dir, sliding_training_sample_file, it_path, batch_size, train_softmax_dir):
        self.unit_size = 16
        self.feats_dimen = 4096
        self.context_num = 1
        self.context_size = 128
        self.visual_feature_dim = 4096*3
        self.sent_vec_dim = 4800
        self.clip_softmax_dim = 400
        self.softmax_unit_size = 32
        self.spacy_vec_dim = 300
        self.train_softmax_dir = train_softmax_dir 
        self.index_in_epoch=0
        self.epochs_completed =0
        self.counter=0
        self.stage_1_iter=5000
        self.batch_size=batch_size
        print "Reading training data list from "+it_path
        cs=pickle.load(open(it_path))
        self.clip_sentence_pairs=cs

        movie_names_set=set()
        self.movie_clip_names={}
        for k in range(len(self.clip_sentence_pairs)):
            clip_name=self.clip_sentence_pairs[k][0]
            movie_name=clip_name.split(" ")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name]=[]
            self.movie_clip_names[movie_name].append(k)
        self.movie_names=list(movie_names_set)
        self.num_samples=len(self.clip_sentence_pairs)
        print str(len(self.clip_sentence_pairs))+" clip-sentence pairs are readed"
        
        self.sliding_clip_path=sliding_dir
        self.clip_sentence_pairs_iou=pickle.load(open(sliding_training_sample_file))
        self.num_videos = len(self.clip_sentence_pairs_iou)

        # get the number of self.clip_sentence_pairs_iou
        self.num_samples_iou = 0
        for ii in self.clip_sentence_pairs_iou:
            for iii in self.clip_sentence_pairs_iou[ii]:
                self.num_samples_iou += len(self.clip_sentence_pairs_iou[ii][iii])
        print self.num_samples_iou, "iou clip-sentence pairs are readed"
 

       # print self.clip_sentence_pairs_iou
        self.movie_length_dict={}
        with open("./ref_info/charades_movie_length_info.txt")  as f:
            for l in f:
                self.movie_length_dict[l.rstrip().split(" ")[0]]=float(l.rstrip().split(" ")[1])

        # get the video name list
        self.v_name_lst = []
        for ii in self.clip_sentence_pairs_iou:
            self.v_name_lst.append(ii)
        
        # get the clip name dict
        self.c_name_dict = {}
        for ii in self.v_name_lst:
            self.c_name_dict[ii] = []
            for iii in self.clip_sentence_pairs_iou[ii]:
                self.c_name_dict[ii].append(iii)
        #print self.c_name_dict
                
            

    '''
    read unit level feats by just passing the start and end number
    '''
    def read_unit_level_feats(self, clip_name):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])
        num_units = (end - start)/self.unit_size
        curr_start = start

        start_end_list = []
        while (curr_start+self.unit_size <= end):
            start_end_list.append((curr_start, curr_start+self.unit_size))
            curr_start += self.unit_size

        original_feats = np.zeros([num_units, self.feats_dimen], dtype=np.float32)
        for k, (curr_s, curr_e) in enumerate(start_end_list):
            one_feat = np.load(self.sliding_clip_path+movie_name+"_"+str(curr_s)+".0_"+str(curr_e)+".0.npy")
            original_feats[k] = one_feat

        return np.mean(original_feats, axis=0)



    '''
    read unit level softmax by just passing the start and end number
    this is also work for softmax with other self.softmax_unit_size, (e.g. 16), nut you should check the code.

    '''
    def read_unit_level_softmax(self, clip_name):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])
        num_units = (end - start)/self.unit_size - (self.softmax_unit_size/self.unit_size) +1
        _is_clip_shorter_than_unit_size = False
        if num_units <= 0:
            num_units = 1
            _is_clip_shorter_than_unit_size = True


        softmax_feats = np.zeros([num_units, self.clip_softmax_dim], dtype=np.float32)
        if _is_clip_shorter_than_unit_size:
            _start_here = start
            _end_here = end
            _npy_file_path_this = self.train_softmax_dir+movie_name+".mp4_"+str(curr_s)+"_"+str(curr_e)+".npy"
            if not os.path.exists(_npy_file_path_this):
                _npy_file_path_this = self.train_softmax_dir+movie_name+".mp4_"+str(curr_s)+"_"+str(curr_e)+".npy"
            one_feat = np.load(_npy_file_path_this)
            softmax_feats[0] = one_feat

        else:
            curr_start = start
            start_end_list = []
            while (curr_start+self.softmax_unit_size <= end):
                start_end_list.append((curr_start, curr_start+self.softmax_unit_size))
                curr_start += self.unit_size
            for k, (curr_s, curr_e) in enumerate(start_end_list):
                one_feat = np.load(self.train_softmax_dir+movie_name+".mp4_"+str(curr_s)+"_"+str(curr_e)+".npy")
                softmax_feats[k] = one_feat

        return np.mean(softmax_feats, axis=0)


    '''
    judge the feats is existed or not
    like os.path.exists(self.sliding_clip_path+left_context_name) in the get_context_window(0)
    '''
    def feat_exists(self, clip_name):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])

        return os.path.exists(self.sliding_clip_path+movie_name+"_"+str(end-16)+".0_"+str(end)+".0.npy") and \
               os.path.exists(self.sliding_clip_path+movie_name+"_"+str(start)+".0_"+str(start+16)+".0.npy")

    '''
    compute left (pre) and right (post) context features based on read_unit_level_feats().
    '''
    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])
        clip_length = self.context_size
        left_context_feats = np.zeros([win_length, self.feats_dimen], dtype=np.float32)
        right_context_feats = np.zeros([win_length, self.feats_dimen], dtype=np.float32)
        last_left_feat = self.read_unit_level_feats(clip_name)
        last_right_feat = self.read_unit_level_feats(clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_"+str(left_context_start)+"_"+str(left_context_end)
            right_context_name = movie_name+"_"+str(right_context_start)+"_"+str(right_context_end)
            if self.feat_exists(left_context_name):
                left_context_feat = self.read_unit_level_feats(left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if self.feat_exists(right_context_name):
                right_context_feat = self.read_unit_level_feats(right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)



    def generate_training_sample_index_one_clip_in_one_video(self):
        """to generate the training smaple index. one video one sample"""
        random_video_index_lst = random.sample(range(self.num_videos), self.batch_size)
        # get the video name list
        lst_video_clip_order_lst = []
        # get the first index
        for ii in random_video_index_lst:
            lst_video_clip_order_lst.append([self.v_name_lst[ii]])
        # get the second index
        for ii in lst_video_clip_order_lst:
            num_clip_here = len(self.c_name_list[ii[0]])
            random_clip_index_here = random.choice(range(num_clip_here))
            ii.append(self.c_name_list[ii[0]][random_clip_index_here])
        # get the third index
        for ii in lst_video_clip_order_lst:
            random_order_index = random.choice(range(len(self.clip_sentence_pairs_iou[ii[0]][ii[1]])))
            ii.append(random_order_index)
    
        return lst_video_clip_order_lst


    def generate_training_sample_index_all_clip_in_one_then_next_one(self):
        """to generate the training smaple index. use all clips in one video then next video"""

        triple_index_lst = []
        cnt_batch = 0
        random_video_index_lst = []
        # get the first and second index
        while cnt_batch < self.batch_size:
            random_video_index = random.choice(range(self.num_videos))
            if random_video_index not in random_video_index_lst:
                video_name = self.v_name_lst[random_video_index]
                random_video_index_lst.append(random_video_index)
                if len(self.clip_sentence_pairs_iou[video_name]) < (self.batch_size - cnt_batch):
                    num_clip_need = len(self.clip_sentence_pairs_iou[video_name])
                else:
                    num_clip_need = (self.batch_size - cnt_batch)
                random_clip_index_lst = random.sample(range(len(self.clip_sentence_pairs_iou[video_name])), num_clip_need)
                for ii in random_clip_index_lst:
                    two_index = []
                    two_index.append(video_name)
                    two_index.append(self.c_name_dict[video_name][ii])
                    triple_index_lst.append(two_index)
                    cnt_batch += 1
        # get the third index
        for ii in triple_index_lst:
            random_order_index = random.choice(range(len(self.clip_sentence_pairs_iou[ii[0]][ii[1]])))
            ii.append(random_order_index)

        return triple_index_lst


    '''
    modified to read dict data
    read next batch of training data, this function is used for training CTRL-reg
    '''
    def next_batch_iou(self):

        image_batch = np.zeros([self.batch_size, self.visual_feature_dim])
        softmax_batch = np.zeros([self.batch_size, self.clip_softmax_dim])
        sentence_batch = np.zeros([self.batch_size, self.sent_vec_dim])
        offset_batch = np.zeros([self.batch_size, 2], dtype=np.float32)
        VP_spacy_batch = np.zeros([self.batch_size, self.spacy_vec_dim*2])
        subj_spacy_batch = np.zeros([self.batch_size, self.spacy_vec_dim])
        obj_spacy_batch = np.zeros([self.batch_size, self.spacy_vec_dim])

        #lst_video_clip_order_lst = self.generate_training_sample_index_one_clip_in_one_video()
        lst_video_clip_order_lst = self.generate_training_sample_index_all_clip_in_one_then_next_one()        

        # read all clips
        for ind_this, index_here in enumerate(lst_video_clip_order_lst):
            # get this clip's: sentence  vector, swin, p_offest, l_offset, sentence, Vps
            dict_3rd = self.clip_sentence_pairs_iou[index_here[0]][index_here[1]][index_here[2]]
            #read visual feats
            featmap = self.read_unit_level_feats(dict_3rd['proposal_or_sliding_window'])
            left_context_feat, right_context_feat = self.get_context_window(dict_3rd['proposal_or_sliding_window'], self.context_num)
            image_batch[ind_this,:] = np.hstack((left_context_feat, featmap, right_context_feat))
            # read softmax batch
            softmax_center_clip = self.read_unit_level_softmax(dict_3rd['proposal_or_sliding_window'])
            softmax_batch[ind_this,:] = softmax_center_clip
            # sentence batch
            sentence_batch[ind_this,:] = dict_3rd['sent_skip_thought_vec'][0][0, :self.sent_vec_dim]
            if len(dict_3rd['dobj_or_VP']) != 0:
                VP_spacy_one_by_one_this_ = dict_3rd['VP_spacy_vec_one_by_one_word'][random.choice(xrange(len(dict_3rd['dobj_or_VP'])))]
                if len(VP_spacy_one_by_one_this_) == 1:
                    VP_spacy_batch[ind_this, :self.spacy_vec_dim] = VP_spacy_one_by_one_this_[0]
                else:
                    VP_spacy_batch[ind_this, :] = np.concatenate((VP_spacy_one_by_one_this_[0], VP_spacy_one_by_one_this_[1]))
            if len(dict_3rd['subj']) != 0:
                subj_spacy_batch[ind_this, :] = dict_3rd['subj_spacy_vec'][random.choice(xrange(len(dict_3rd['subj'])))]
            if len(dict_3rd['obj']) != 0:
                obj_spacy_batch[ind_this, :] = dict_3rd['obj_spacy_vec'][random.choice(xrange(len(dict_3rd['obj'])))]

            # offest
            p_offset = dict_3rd['offset_start']
            l_offset = dict_3rd['offset_end']
            offset_batch[ind_this,0] = p_offset
            offset_batch[ind_this,1] = l_offset

        simi_mat_img = cosine_similarity(image_batch, image_batch)
        np.fill_diagonal(simi_mat_img, 1.0)

        return image_batch, sentence_batch, offset_batch, softmax_batch, VP_spacy_batch, subj_spacy_batch, obj_spacy_batch, simi_mat_img

class TestingDataSet(object):
    def __init__(self, img_dir, csv_path, batch_size, test_swin_txt_path, test_softmax_dir, test_clip_sentence_pairs_path):
        #il_path: image_label_file path
        self.context_num = 1
        self.context_size = 128
        self.visual_feature_dim = 4096*3
        self.feats_dimen = 4096
        self.unit_size = 16
        self.context_size = 128
        self.semantic_size = 4800
        self.sliding_clip_path = img_dir
        self.index_in_epoch=0
        self.spacy_vec_dim = 300
        self.sent_vec_dim = 4800
        self.clip_softmax_dim = 400
        self.softmax_unit_size = 32
        self.test_softmax_dir = test_softmax_dir
        self.epochs_completed =0
        self.batch_size=batch_size
        self.test_swin_txt_path = test_swin_txt_path
        print "Reading testing data list from "+csv_path
        csv=pickle.load(open(csv_path))
        self.num_samples=len(csv)
        self.clip_sentence_pairs = pickle.load(open(test_clip_sentence_pairs_path)) 
        print str(len(self.clip_sentence_pairs))+" test videos are readed"

        movie_names_set = set()
        for ii in self.clip_sentence_pairs:
            for iii in self.clip_sentence_pairs[ii]:
                clip_name = iii
                movie_name = ii
                if not movie_name in movie_names_set:
                    movie_names_set.add(movie_name)
        self.movie_names = list(movie_names_set)
  
        self.sliding_clip_names = []
        with open(self.test_swin_txt_path) as f:
            for l in f:
                self.sliding_clip_names.append(l.rstrip().replace(" ", "_"))
        print "sliding clips number for test: "+str(len(self.sliding_clip_names))
        assert self.batch_size <= self.num_samples

        self.movie_length_dict={}
        with open("./ref_info/charades_movie_length_info.txt")  as f:
            for l in f:
                self.movie_length_dict[l.rstrip().split(" ")[0]]=float(l.rstrip().split(" ")[1])


    '''
    read unit level feats by just passing the start and end number
    '''
    def read_unit_level_feats(self, clip_name):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])
        num_units = (end - start)/self.unit_size
        curr_start = start

        start_end_list = []
        while (curr_start+self.unit_size <= end):
            start_end_list.append((curr_start, curr_start+self.unit_size))
            curr_start += self.unit_size

        original_feats = np.zeros([num_units, self.feats_dimen], dtype=np.float32)
        for k, (curr_s, curr_e) in enumerate(start_end_list):
            one_feat = np.load(self.sliding_clip_path + movie_name+"_"+str(curr_s)+".0_"+str(curr_e)+".0.npy")
            original_feats[k] = one_feat

        return np.mean(original_feats, axis=0)


    '''
    read unit level softmax by just passing the start and end number
    this is also work for softmax with other self.softmax_unit_size, (e.g. 16), nut you should check the code.

    '''
    def read_unit_level_softmax(self, clip_name):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])
        num_units = (end - start)/self.unit_size - (self.softmax_unit_size/self.unit_size) +1
        _is_clip_shorter_than_unit_size = False
        if num_units <= 0:
            num_units = 1
            _is_clip_shorter_than_unit_size = True


        softmax_feats = np.zeros([num_units, self.clip_softmax_dim], dtype=np.float32)
        if _is_clip_shorter_than_unit_size:
            _start_here = start
            _end_here = end
            _npy_file_path_this = self.test_softmax_dir+movie_name+".mp4_"+str(curr_s)+"_"+str(curr_e)+".npy"
            if not os.path.exists(_npy_file_path_this):
                _npy_file_path_this = self.test_softmax_dir+movie_name+".mp4_"+str(curr_s)+"_"+str(curr_e)+".npy"
            one_feat = np.load(_npy_file_path_this)
            softmax_feats[0] = one_feat

        else:
            curr_start = start
            start_end_list = []
            while (curr_start+self.softmax_unit_size <= end):
                start_end_list.append((curr_start, curr_start+self.softmax_unit_size))
                curr_start += self.unit_size
            for k, (curr_s, curr_e) in enumerate(start_end_list):
                one_feat = np.load(self.test_softmax_dir+movie_name+".mp4_"+str(curr_s)+"_"+str(curr_e)+".npy")
                softmax_feats[k] = one_feat

        return np.mean(softmax_feats, axis=0)



    '''
    judge the feats is existed or not
    like os.path.exists(self.sliding_clip_path+left_context_name) in the get_context_window(0)
    '''
    def feat_exists(self, clip_name):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])

        return os.path.exists(self.sliding_clip_path+movie_name+"_"+str(end-16)+".0_"+str(end)+".0.npy") and \
               os.path.exists(self.sliding_clip_path+movie_name+"_"+str(start)+".0_"+str(start+16)+".0.npy")





    '''
    compute left (pre) and right (post) context features based on read_unit_level_feats().
    '''
    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])
        clip_length = self.context_size
        left_context_feats = np.zeros([win_length, self.feats_dimen], dtype=np.float32)
        right_context_feats = np.zeros([win_length, self.feats_dimen], dtype=np.float32)
        last_left_feat = self.read_unit_level_feats(clip_name)
        last_right_feat = self.read_unit_level_feats(clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_"+str(left_context_start)+"_"+str(left_context_end)
            right_context_name = movie_name+"_"+str(right_context_start)+"_"+str(right_context_end)
            if self.feat_exists(left_context_name):
                left_context_feat = self.read_unit_level_feats(left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if self.feat_exists(right_context_name):
                right_context_feat = self.read_unit_level_feats(right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)



    '''
    load unit level feats and sentence vector
    '''
    def load_movie_slidingclip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_featmap = []

        for dict_2nd in self.clip_sentence_pairs[movie_name]:
            for dict_3rd in self.clip_sentence_pairs[movie_name][dict_2nd]:

                VP_spacy_vec_ = np.zeros(self.spacy_vec_dim*2)
                subj_spacy_vec_ = np.zeros(self.spacy_vec_dim)
                obj_spacy_vec_ = np.zeros(self.spacy_vec_dim)

                if len(dict_3rd['dobj_or_VP']) != 0:
                    VP_spacy_one_by_one_this_ = dict_3rd['VP_spacy_vec_one_by_one_word'][random.choice(xrange(len(dict_3rd['dobj_or_VP'])))]
                    if len(VP_spacy_one_by_one_this_) == 1:
                        VP_spacy_vec_[:self.spacy_vec_dim] = VP_spacy_one_by_one_this_[0]
                    else:
                        VP_spacy_vec_ = np.concatenate((VP_spacy_one_by_one_this_[0], VP_spacy_one_by_one_this_[1]))
                if len(dict_3rd['subj']) != 0:
                    subj_spacy_vec_ = dict_3rd['subj_spacy_vec'][random.choice(xrange(len(dict_3rd['subj'])))]
                if len(dict_3rd['obj']) != 0:
                    obj_spacy_vec_ = dict_3rd['obj_spacy_vec'][random.choice(xrange(len(dict_3rd['obj'])))]


                sentence_vec_  = dict_3rd['sent_skip_thought_vec'][0][0, :self.sent_vec_dim]

                movie_clip_sentences.append((dict_2nd, sentence_vec_, VP_spacy_vec_, subj_spacy_vec_, obj_spacy_vec_))

        for k in xrange(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:
                left_context_feat,right_context_feat = self.get_context_window(self.sliding_clip_names[k], self.context_num)
                feature_data = self.read_unit_level_feats(self.sliding_clip_names[k])

                # read softmax batch
                softmax_center_clip = self.read_unit_level_softmax(self.sliding_clip_names[k])

                comb_feat = np.hstack((left_context_feat, feature_data, right_context_feat))
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat, softmax_center_clip))
                #movie_clip_featmap.append((self.sliding_clip_na
        return movie_clip_featmap, movie_clip_sentences



