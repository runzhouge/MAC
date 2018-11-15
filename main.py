from __future__ import division
import tensorflow as tf
import numpy as np
import acl_model
from six.moves import xrange
import time
from sklearn.metrics import average_precision_score
import pickle
import mpu
import operator
import os
import argparse
import math
from scipy.special import expit




#=====================================================================
GPU_MEM_FRACTION = 0.42
TEST_SAVE_EVERY = 2000
MAX_TRAIN_STEP = 20005
BATCH_SIZE_TRAIN = 28
SEED_ = 1234
#=====================================================================




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='TALL next')
    parser.add_argument('--is_only_test', dest='is_only_test', help='If it is only for test without trianing, use True',
        default=False)
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='The path to read the checkpoint when test',
        default='./trained_save/')
    parser.add_argument('--test_name', dest='test_name', help='The test name which will be displayed in the test txt file',
        default='one_time_test')
    parser.add_argument('--save_checkpoint_parent_dir', dest='save_checkpoint_parent_dir', help='the parent folder to save the trained model in training',
        default='./trained_save/')
    parser.add_argument('--is_continue_training', dest='is_continue_training', help='If this is for continuing training the model, use True',
        default=False)
    parser.add_argument('--checkpoint_path_continue_training', dest='checkpoint_path_continue_training', help='The path to read the checkpoint when continue training ',
        default='./trained_save/')
    args = parser.parse_args()

    return args



def compute_recall_top_n(top_n,class_score_matrix,labels):
    correct_num=0.0
    for k in range(class_score_matrix.shape[0]):
        class_score=class_score_matrix[k,:]
        predictions=class_score.argsort()[::-1][0:top_n]
        if labels[k] in predictions: correct_num+=1
    return correct_num, correct_num/len(labels)

def compute_precision_top_n(top_n,sentence_image_mat,sclips,iclips):
    correct_num=0.0
    for k in range(sentence_image_mat.shape[0]):
        gt=sclips[k]
        sim_v=sentence_image_mat[k]
        sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        for index in sim_argsort:
            if gt == iclips[index]:
                correct_num+=1
    return correct_num

def calculate_IoU(i0,i1):
    union=(min(i0[0],i1[0]) , max(i0[1],i1[1]))
    inter=(max(i0[0],i1[0]) , min(i0[1],i1[1]))
    iou=1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick

    union = map(operator.sub, x2, x1) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

def compute_IoU_recall_top_n(top_n,iou_thresh,sentence_image_mat,sclips,iclips):
    correct_num=0.0
    for k in range(sentence_image_mat.shape[0]):
        gt=sclips[k]
        gt_start=float(gt.split(" ")[1])
        gt_end=float(gt.split(" ")[2])
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        sim_v=[v for v in sentence_image_mat[k]]
        starts=[float(iclip.split("_")[1]) for iclip in iclips]
        ends=[float(iclip.split("_")[2]) for iclip in iclips]
        picks=nms_temporal(starts,ends,sim_v,iou_thresh-0.05)
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n<len(picks): picks=picks[0:top_n]
        for index in picks:
            pred_start=float(iclips[index].split("_")[1])
            pred_end=float(iclips[index].split("_")[2])
            iou=calculate_IoU((gt_start,gt_end),(pred_start,pred_end))
            if iou>=iou_thresh:    
                correct_num+=1
                break
    return correct_num


def compute_IoU_recall_top_n_forreg(top_n,iou_thresh,sentence_image_mat,sentence_image_reg_mat,sclips,iclips):
    correct_num=0.0
    for k in range(sentence_image_mat.shape[0]):
        gt=sclips[k]
        gt_start=float(gt.split("_")[1])
        gt_end=float(gt.split("_")[2])
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        sim_v=[v for v in sentence_image_mat[k]]
        starts=[s for s in sentence_image_reg_mat[k,:,0]]
        ends=[e for e in sentence_image_reg_mat[k,:,1]]
        picks=nms_temporal(starts,ends,sim_v,iou_thresh-0.05)
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n<len(picks): picks=picks[0:top_n]
        for index in picks:
            pred_start=sentence_image_reg_mat[k,index,0]
            pred_end=sentence_image_reg_mat[k,index,1]
            iou=calculate_IoU((gt_start,gt_end),(pred_start,pred_end))
            if iou>=iou_thresh:
                correct_num+=1
                break
    return correct_num

def do_eval_slidingclips(sess,vs_eval_op,model,movie_length_info,iter_step, test_result_output):
    IoU_thresh=[0.5, 0.7]
    all_correct_num_10=[0.0]*5
    all_correct_num_5=[0.0]*5
    all_correct_num_1=[0.0]*5
    all_retrievd=0.0
    for movie_name in model.test_set.movie_names:
        movie_length=movie_length_info[movie_name]
        #print "Test movie: "+movie_name+"....loading movie data"
        movie_clip_featmaps, movie_clip_sentences=model.test_set.load_movie_slidingclip(movie_name,16)
        #print "sentences: "+ str(len(movie_clip_sentences))
        #print "clips: "+ str(len(movie_clip_featmaps))
        sentence_image_mat=np.zeros([len(movie_clip_sentences),len(movie_clip_featmaps)])
        sentence_image_reg_mat=np.zeros([len(movie_clip_sentences),len(movie_clip_featmaps),2])
        for k in range(len(movie_clip_sentences)):

            sent_vec=movie_clip_sentences[k][1]
            VP_spacy_vec = movie_clip_sentences[k][2]
            subj_spacy_vec = movie_clip_sentences[k][3]
            obj_spacy_vec = movie_clip_sentences[k][4]

            sent_vec=np.reshape(sent_vec,[1,sent_vec.shape[0]])
            VP_spacy_vec=np.reshape(VP_spacy_vec,[1,VP_spacy_vec.shape[0]])
            subj_spacy_vec=np.reshape(subj_spacy_vec,[1,subj_spacy_vec.shape[0]])
            obj_spacy_vec=np.reshape(obj_spacy_vec,[1,obj_spacy_vec.shape[0]])


            for t in range(len(movie_clip_featmaps)):


                featmap = movie_clip_featmaps[t][1]
                softmax_ = movie_clip_featmaps[t][2]
                visual_clip_name = movie_clip_featmaps[t][0]
                
                # the contents of visual_clip_name
                # 0: name; 1: swin_start; 2:swin_end; 3: round_reg_start;
                # 4: round_reg_end; 5: reg_start; 6:reg_end; 7: score; others

                # swin_start and swin_end
                start=float(visual_clip_name.split("_")[1])
                end=float(visual_clip_name.split("_")[2])
                conf_score = float(visual_clip_name.split("_")[7])

                featmap=np.reshape(featmap,[1,featmap.shape[0]])
                softmax_ = np.reshape(softmax_, [1, softmax_.shape[0]])

                feed_dict = {
                model.visual_featmap_ph_test: featmap,
                model.sentence_ph_test:sent_vec,
                model.VP_spacy_ph_test:VP_spacy_vec,
                model.softmax_ph_test: softmax_

                }
                outputs=sess.run(vs_eval_op,feed_dict=feed_dict)
                
                sentence_image_mat[k,t] = expit(outputs[0]) * conf_score

                reg_end=end+outputs[2]
                reg_start=start+outputs[1]
                sentence_image_reg_mat[k,t,0]=reg_start
                sentence_image_reg_mat[k,t,1]=reg_end
                
        iclips=[b[0] for b in movie_clip_featmaps]
        sclips=[b[0] for b in movie_clip_sentences]
        
        for k in range(len(IoU_thresh)):
            IoU=IoU_thresh[k]
            correct_num_10=compute_IoU_recall_top_n_forreg(10,IoU,sentence_image_mat,sentence_image_reg_mat,sclips,iclips)
            correct_num_5=compute_IoU_recall_top_n_forreg(5,IoU,sentence_image_mat,sentence_image_reg_mat,sclips,iclips)
            correct_num_1=compute_IoU_recall_top_n_forreg(1,IoU,sentence_image_mat,sentence_image_reg_mat,sclips,iclips)
            all_correct_num_10[k]+=correct_num_10
            all_correct_num_5[k]+=correct_num_5
            all_correct_num_1[k]+=correct_num_1
        all_retrievd+=len(sclips)
    for k in range(len(IoU_thresh)):
        print " IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd)

        test_result_output.write("Step "+str(iter_step)+": IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd)+"\n")



def run_training():
    max_steps_stage1=0
    max_steps=MAX_TRAIN_STEP
    batch_size=BATCH_SIZE_TRAIN
    home_dir = os.environ['HOME']


    # visual feature for each video clip
    # the fc6 layer outputs of C3D pretraiend on Sprots-1M, 4096-dimension
    train_feature_dir = "/change/directory/to/all_fc6_unit16_overlap0.5/"
    test_feature_dir = "/change/directory/to/all_fc6_unit16_overlap0.5/"


    # visual activity concepts for each video clip
    # the softmax layer ouput of R(2+1)D pretraiend on Kinetics, 400-demension
    train_softmax_dir = '/change/directory/to/train_softmax/'
    test_softmax_dir = '/change/directory/to/test_softmax/'






    # clip ground truth and corresponding sentence embedding
    # this is originally from https://github.com/jiyanggao/TALL
    # we didn't use the setence embdedding provided in these files
    # we use our own sentence embedding which will appear later
    train_csv_path = "./ref_info/charades_sta_train_clip-sentvec_o0.5_l10_activity_nodup.pkl"
    test_csv_path = "./ref_info/charades_sta_test_clip-sentvec_o0.5_l10_activity_nodup.pkl"    
  

 
    # sentece embedding, verb-object vector
    
    # the content of each item is organized as dict --> dict --> lst --> dict
    # i.e. The first level dict is the dict for different videos
    # the second level dict is the dict  for different video clips in one video
    # the third level list is the different sliding windows and different sentence for one video clip
    # the fourth level dict contains the components we use, e.g. skip-thought sentence embedding, glove word vector and so on.
    # please refer to the paper and correponding pkl file for more details.
 
    # sentece embedding is extracted by skip-thoughts (4800-dimesnion)
    # verb-object vector is extracted by stanford glove (300-dimension)
    train_clip_sentence_pairs_iou_path = "./ref_info/charades_sta_train_semantic_sentence_VP_sub_obj.pkl" 
    test_clip_sentence_pairs_path = './ref_info/charades_sta_test_semantic_sentence_VP_sub_obj.pkl'
    
    # the propsal score used in test
    # trained on TURN TAP, https://github.com/jiyanggao/TURN-TAP
    # the contents of each line are:
    # 0: name; 1: swin_start; 2:swin_end; 3: round_reg_start;
    # 4: round_reg_end; 5: reg_start; 6:reg_end; 7: proposal_confident_score; 
    # 8: others; 9: others
    test_swin_txt_path = "./ref_info/charades_sta_test_swin_props_num_36364.txt"


    # arguments
    args = parse_args()
    is_only_test = args.is_only_test
    checkpoint_path = args.checkpoint_path
    save_checkpoint_parent_dir = args.save_checkpoint_parent_dir
    is_continue_training = args.is_continue_training
    checkpoint_path_continue_training = args.checkpoint_path_continue_training
    test_name = args.test_name



    model=acl_model.acl_model(batch_size,train_csv_path, test_csv_path, test_feature_dir, train_feature_dir, train_clip_sentence_pairs_iou_path, test_clip_sentence_pairs_path, test_swin_txt_path, train_softmax_dir, test_softmax_dir)


    # if it is test only
    if is_only_test:

        # txt file to save the test results
        localtime = time.asctime(time.localtime(time.time()))
        _localtime = localtime.replace(" ", "_").replace(":", "_")
        txt_dir = './results_history/'
        txt_dir = txt_dir + 'ctrl_test_results_' + _localtime+ '_only_one_test.txt'
        test_result_output=open(txt_dir, "w")

        with tf.Graph().as_default():
            tf.set_random_seed(SEED_)
            loss_align_reg,vs_train_op,vs_eval_op,offset_pred,loss_reg=model.construct_model()
            # Create a session for running Ops on the Graph.
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_MEM_FRACTION)
            sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
            saver = tf.train.Saver(max_to_keep=1000)
            merged=tf.summary.merge_all()
            writer=tf.summary.FileWriter("./tf_summary/",sess.graph_def)
            saver.restore(sess, checkpoint_path)
            print "Model restored from " + checkpoint_path, "----------------------------------\n"
            movie_length_dict={}
            with open("./ref_info/charades_movie_length_info.txt") as f:
                for l in f:
                    movie_length_dict[l.rstrip().split(" ")[0]]=float(l.rstrip().split(" ")[2])
            
            print "Start to test:-----------------\n"
            do_eval_slidingclips(sess,vs_eval_op,model,movie_length_dict,test_name, test_result_output)
    else:

        # txt file to save the test results
        localtime = time.asctime(time.localtime(time.time()))
        _localtime = localtime.replace(" ", "_").replace(":", "_")
        txt_dir = './results_history/'
        txt_dir = txt_dir + 'ctrl_test_results_' + _localtime+ '.txt'
        test_result_output=open(txt_dir, "w")


        # folder to save the trained model
        if not os.path.exists(save_checkpoint_parent_dir+_localtime):
            os.makedirs(save_checkpoint_parent_dir+_localtime)

        # if it is continuing training
        if is_continue_training:
            # if there is a error in this line, please assign the cuurent_step yourself.
            current_step = int(checkpoint_path_continue_training.split("-")[-1])
            with tf.Graph().as_default():
                tf.set_random_seed(SEED_)
                loss_align_reg,vs_train_op,vs_eval_op,offset_pred,loss_reg=model.construct_model()
                # Create a session for running Ops on the Graph.
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_MEM_FRACTION)
                sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
                saver = tf.train.Saver(max_to_keep=1000)
                merged=tf.summary.merge_all()
                writer=tf.summary.FileWriter("./tf_summary/",sess.graph_def)
                saver.restore(sess, checkpoint_path_continue_training)
                print "Model restored from " + checkpoint_path, "----------------------------------\n"

                for step in xrange(current_step, max_steps):
                    
                    start_time = time.time()
                    feed_dict = model.fill_feed_dict_train_reg()
                    _,loss_value,offset_pred_v,loss_reg_v = sess.run([vs_train_op,loss_align_reg,offset_pred,loss_reg], feed_dict=feed_dict)
                    duration = time.time() - start_time
                    if step % 5 == 0:
                        # Print status to stdout.
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                        #print loss_reg_v
                        #writer.add_summary(sum_str,step)


                    if (step+1) % TEST_SAVE_EVERY == 0:
                     
                        save_path = saver.save(sess, save_checkpoint_parent_dir+_localtime+"/trained_model.ckpt", global_step=step+1)
                        print "Model saved to " + save_path, "----------------------------------\n"

                        print "Start to test:-----------------\n"
                        movie_length_dict={}
                        with open("./ref_info/charades_movie_length_info.txt") as f:
                            for l in f:
                                movie_length_dict[l.rstrip().split(" ")[0]]=float(l.rstrip().split(" ")[2])

                        do_eval_slidingclips(sess,vs_eval_op,model,movie_length_dict,step+1, test_result_output)


        
        else:
            current_step = 0

            with tf.Graph().as_default():
                tf.set_random_seed(SEED_)
                loss_align_reg,vs_train_op,vs_eval_op,offset_pred,loss_reg=model.construct_model()
                # Create a session for running Ops on the Graph.
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_MEM_FRACTION)
                sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
                saver = tf.train.Saver(max_to_keep=1000)
                merged=tf.summary.merge_all()
                writer=tf.summary.FileWriter("./tf_summary/",sess.graph_def)
                # Run the Op to initialize the variables.
                init = tf.global_variables_initializer()
                sess.run(init)
                for step in xrange(current_step, max_steps):
                    
                    start_time = time.time()
                    feed_dict = model.fill_feed_dict_train_reg()
                    _,loss_value,offset_pred_v,loss_reg_v = sess.run([vs_train_op,loss_align_reg,offset_pred,loss_reg], feed_dict=feed_dict)
                    duration = time.time() - start_time
                    if step % 5 == 0:
                        # Print status to stdout.
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                        #print loss_reg_v
                        #writer.add_summary(sum_str,step)


                    if (step+1) % TEST_SAVE_EVERY == 0:
                     
                        save_path = saver.save(sess, save_checkpoint_parent_dir+_localtime+"/trained_model.ckpt", global_step=step+1)
                        print "Model saved to " + save_path, "----------------------------------\n"

                        print "Start to test:-----------------\n"
                        movie_length_dict={}
                        with open("./ref_info/charades_movie_length_info.txt") as f:
                            for l in f:
                                movie_length_dict[l.rstrip().split(" ")[0]]=float(l.rstrip().split(" ")[2])

                        do_eval_slidingclips(sess,vs_eval_op,model,movie_length_dict,step+1, test_result_output)



    
def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
        	



