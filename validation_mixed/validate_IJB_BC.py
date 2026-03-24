import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))

from ..validation_mixed.insightface_ijb_helper.dataloader import prepare_dataloader
from ..validation_mixed.insightface_ijb_helper import eval_helper_identification
from ..validation_mixed.insightface_ijb_helper import eval_helper as eval_helper_verification

from ..expert import gabor
from ..utils import kernel_pca

import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
import pandas as pd


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms):

    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    if stacked_norms is not None:
        assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)

    pre_norm_embeddings = stacked_embeddings * stacked_norms
    fused = pre_norm_embeddings.sum(dim=0)
    fused, fused_norm = l2_norm(fused, axis=1)

    return fused, fused_norm

def infer_images(model, img_root, landmark_list_path, batch_size, use_flip_test, device, expert=False):
    img_list = open(landmark_list_path)
    # img_aligner = ImageAligner(image_size=(112, 112))

    files = img_list.readlines()
    print('files:', len(files))
    
    faceness_scores = [] # điểm chất lượng khuôn mặt
    img_paths = []
    landmarks = []
    for img_index, each_line in enumerate(files):
        name_lmk_score = each_line.strip().split(' ')
        img_path = os.path.join(img_root, name_lmk_score[0])
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        img_paths.append(img_path)
        landmarks.append(lmk)
        faceness_scores.append(name_lmk_score[-1])

    print('total images : {}'.format(len(img_paths)))
    
    dataloader = prepare_dataloader(img_paths, landmarks, batch_size, num_workers=0, image_size=(112, 112)) # DataLoader
    model.eval()
    features = []
    norms = []
    
    all_features = []
    if expert:
        bank = gabor.build_gabor_bank()
        print(f'Gabor bank built with {len(bank)} filters')
    
    with torch.no_grad():
        for images, idx in tqdm(dataloader):
            
            # GABOR FILTERS APPLIED HERE
            if expert:
                images_ = images.permute(0, 2, 3, 1)
                grays = [rgb2gray(img) for img in images_]
                list_responses = [gabor.apply_gabor_bank(gray, bank) for gray in grays]
                expert_features = []
                for response in list_responses:
                    images_ = []
                    for entry in response:
                        images_.append(entry['magnitude'].flatten())
                    expert_features.append(np.concatenate(images_, axis=0)) # (32, 225792)
                expert_features = torch.tensor(expert_features, dtype=torch.float32)
                all_features.append(expert_features.numpy()) # (N, 32, _)
            
            
            # GABOR FILTERS APPLIED HERE
            feature = model(images.to(device)) # (32, 512)
            if isinstance(feature, tuple):
                feature, norm = feature
            else:
                norm = None

            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images.to(device))
                if isinstance(flipped_feature, tuple):
                    flipped_feature, flipped_norm = flipped_feature
                else:
                    flipped_norm = None

                stacked_embeddings = torch.stack([feature, flipped_feature], dim=0) # (2, B, D)
                if norm is not None:
                    stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                else:
                    staked_norms = None

                fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, stacked_norms)
                features.append(fused_feature.cpu().numpy())
                norms.append(fused_norm.cpu().numpy())
            else:
                features.append(feature.cpu().numpy()) # (N, 32, 512)
                norms.append(norm.cpu().numpy())
    
    # Apply Kernel PCA
    if expert:
        print('Start to apply kernel PCA on Gabor features\n')
        print(f"Features shape before PCA: ({len(all_features)},{len(all_features[0][0])})")
        kpca, all_features = kernel_pca(torch.cat(all_features, dim=0).numpy(), n_components=512) # (N, 512)
        print(f"Features shape after PCA: {all_features.shape}\n")
        # features: (N, 32, 512) -> List numpy
        # all_features: (N, 32, 512) -> List numpy
        combined_features = np.concatenate([features, all_features], axis=2) # (N, 32, 1024)
        img_feats = np.concatenate(list(combined_features), axis=0).astype(np.float32) # (N, 1024)
        
        print(f"Embeddings shape after concatenating with Gabor features: {embeddings.shape}\n")
    else:
        features = np.concatenate(features, axis=0) # (N, 512)
        img_feats = np.array(features).astype(np.float32)
        
       
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    norms = np.concatenate(norms, axis=0)

    assert len(features) == len(img_paths)

    return img_feats, faceness_scores, norms

def identification(data_root, dataset_name, img_input_feats, save_path):

    # Step1: Load Meta Data
    meta_dir = os.path.join(data_root, dataset_name, 'meta')
    if dataset_name == 'IJBC':
        gallery_s1_record = "%s_1N_gallery_G1.csv" % (dataset_name.lower())
        gallery_s2_record = "%s_1N_gallery_G2.csv" % (dataset_name.lower())
    else:
        gallery_s1_record = "%s_1N_gallery_S1.csv" % (dataset_name.lower())
        gallery_s2_record = "%s_1N_gallery_S2.csv" % (dataset_name.lower())
        
    gallery_s1_templates, gallery_s1_subject_ids = eval_helper_identification.read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s1_record))
    print(gallery_s1_templates.shape, gallery_s1_subject_ids.shape)

    gallery_s2_templates, gallery_s2_subject_ids = eval_helper_identification.read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s2_record))
    print(gallery_s2_templates.shape, gallery_s2_templates.shape)

    gallery_templates = np.concatenate(
        [gallery_s1_templates, gallery_s2_templates])
    gallery_subject_ids = np.concatenate(
        [gallery_s1_subject_ids, gallery_s2_subject_ids])
    print(gallery_templates.shape, gallery_subject_ids.shape)

    media_record = "%s_face_tid_mid.txt" % dataset_name.lower()
    total_templates, total_medias = eval_helper_identification.read_template_media_list(
        os.path.join(meta_dir, media_record))
    print("total_templates", total_templates.shape, total_medias.shape)

    # # Step2: Get gallery Features
    gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = eval_helper_identification.image2template_feature(
        img_input_feats, total_templates, total_medias, gallery_templates, gallery_subject_ids)
    print("gallery_templates_feature", gallery_templates_feature.shape)
    print("gallery_unique_subject_ids", gallery_unique_subject_ids.shape)

    # # step 4 get probe features
    probe_mixed_record = "%s_1N_probe_mixed.csv" % dataset_name.lower()
    probe_mixed_templates, probe_mixed_subject_ids = eval_helper_identification.read_template_subject_id_list(
        os.path.join(meta_dir, probe_mixed_record))
    print(probe_mixed_templates.shape, probe_mixed_subject_ids.shape)
    probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = eval_helper_identification.image2template_feature(
        img_input_feats, total_templates, total_medias, probe_mixed_templates,
        probe_mixed_subject_ids)
    print("probe_mixed_templates_feature", probe_mixed_templates_feature.shape)
    print("probe_mixed_unique_subject_ids",
          probe_mixed_unique_subject_ids.shape)

    gallery_ids = gallery_unique_subject_ids
    gallery_feats = gallery_templates_feature
    probe_ids = probe_mixed_unique_subject_ids
    probe_feats = probe_mixed_templates_feature

    mask = eval_helper_identification.gen_mask(probe_ids, gallery_ids)
    identification_result = eval_helper_identification.evaluation(probe_feats, gallery_feats, mask)
    pd.DataFrame(identification_result, index=['identification']).to_csv(os.path.join(save_path, "identification_result.csv"))
    
def verification(data_root, dataset_name, img_input_feats, save_path):
    templates, medias = eval_helper_verification.read_template_media_list(
        os.path.join(data_root, '%s/meta' % dataset_name, '%s_face_tid_mid.txt' % dataset_name.lower()))
    p1, p2, label = eval_helper_verification.read_template_pair_list(
        os.path.join(data_root, '%s/meta' % dataset_name,
                    '%s_template_pair_label.txt' % dataset_name.lower()))

    template_norm_feats, unique_templates = eval_helper_verification.image2template_feature(img_input_feats, templates, medias)
    score = eval_helper_verification.verification(template_norm_feats, unique_templates, p1, p2)

    # # Step 5: Get ROC Curves and TPR@FPR Table
    score_save_file = os.path.join(save_path, "verification_score.npy")
    np.save(score_save_file, score)
    result_files = [score_save_file]
    eval_helper_verification.write_result(result_files, save_path, dataset_name, label)
    os.remove(score_save_file)



def get_features(root, model, model_name, dataset_name, batch_size,
            device, use_flip_test=True):

    save_path = './result/{}/{}'.format(dataset_name, model_name)
    print('result save_path', save_path)
    os.makedirs(save_path, exist_ok=True)

    # get features and fuse
    img_root = os.path.join(root, dataset_name, 'loose_crop')
    landmark_list_path = os.path.join(root, f'{dataset_name}/meta/{dataset_name.lower()}_name_5pts_score.txt')
    img_input_feats, faceness_scores, norms = infer_images(model=model,
                                                           img_root=img_root,
                                                           landmark_list_path=landmark_list_path,
                                                           batch_size=batch_size,
                                                           use_flip_test=use_flip_test,
                                                           device=device)

    print('Feature Shape: ({} , {}) .'.format(img_input_feats.shape[0], img_input_feats.shape[1]))

    
    return img_input_feats * norms, save_path


def evaluate(root, dataset_name, img_input_feats, save_path):
    identification(root, dataset_name, img_input_feats, save_path)
    verification(root, dataset_name, img_input_feats, save_path)