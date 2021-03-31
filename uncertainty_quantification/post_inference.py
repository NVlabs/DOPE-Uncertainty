"""
This file quantifies uncertainty after running the DOPE inference code
"""

import numpy as np
import visii
import simplejson as json
import matplotlib.pyplot as plt
from scipy import stats
from add_simple import get_add, create_obj
from fitting import GP, LR
import argparse
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data",
        default="output/hope_result/results/result_AlphabetSoup/raw_data.json",
        help='folder for the inference configs')
    parser.add_argument("--object",
        default="Ketchup",
        help='which object for ensemble method')
    parser.add_argument("--resultf",
        default="output/hope_result/results/result_AlphabetSoup",
        help='folder to save statistical results')
    parser.add_argument("--model",
        default=None,
        nargs='+',        
        help='models used in inference')
    parser.add_argument('--ensemble',
        action='store_true',
        default=True,
        help = "use ensemble method")
    parser.add_argument("--real_world",
        action='store_true',
        default=False,
        help='real_world setting does not need ground truth')

    opt = parser.parse_args()

    # load the raw data json file
    with open(opt.raw_data, 'r') as f:
      all_data = json.load(f)
    
    greedy_choice = {}
    corr_summary = {}
    Names = []
    all_score = {}
    all_pose_std = {}
    all_location = {}
    all_orientation = {}
    if not opt.real_world:
        all_add = {}

    for m in opt.model:
        greedy_choice[m] = {}
        corr_summary[m] = {}
        all_score[m] = []
        all_pose_std[m] = []
        all_location[m] = []
        all_orientation[m] = []
        if not opt.real_world:
            all_add[m] = []

    if os.path.isdir(opt.resultf):
        pass
    else:
        os.makedirs(opt.resultf)

    for img in all_data:
        Names.append(img)
        data = all_data[img]
        for m in opt.model:
            all_score[m].append(data[m]['score'])
            all_pose_std[m].append(data[m]['GUAPO std'])
            all_location[m].append(data[m]['loc'])
            all_orientation[m].append(data[m]['quat'])
            if not opt.real_world:
                all_add[m].append(data[m]['add error'])

    for m in opt.model:
        all_score[m] = np.array(all_score[m])
        all_pose_std[m] = np.array(all_pose_std[m])
        if not opt.real_world:
            all_add[m] = np.array(all_add[m])

    # greedy choice from score
    best_index_score = {}
    for m in all_score:
        best_index_score[m] = 0
        for index in range(len(all_score[m])):
            if all_score[m][index] > all_score[m][best_index_score[m]]:
                best_index_score[m] = index
        print(m+' (score) best image is', Names[best_index_score[m]])
        print(m+' (score) best score is', all_score[m][best_index_score[m]])
        if not opt.real_world:
            print(m+' (score) add error is', all_add[m][best_index_score[m]])
        if all_score[m][best_index_score[m]] > 0:
            greedy_choice[m]['(score) best frame'] = Names[best_index_score[m]]
            greedy_choice[m]['(score) best score'] = str(all_score[m][best_index_score[m]])
            if not opt.real_world:
                greedy_choice[m]['(score) add error'] = str(all_add[m][best_index_score[m]])
        
    # greedy choice from GUAPO
    for m in all_pose_std:
        best_index = 0
        for index in range(len(all_pose_std[m])):
            if all_pose_std[m][index] < all_pose_std[m][best_index]:
                best_index = index
        print(m+' (GUAPO) best image is', Names[best_index])
        print(m+' (GUAPO) best std is', all_pose_std[m][best_index])
        if not opt.real_world:
            print(m+' (GUAPO) add error is', all_add[m][best_index])
        if all_pose_std[m][best_index] < 1000:
            greedy_choice[m]['(GUAPO) best frame'] = Names[best_index]
            greedy_choice[m]['(GUAPO) best std'] = str(all_pose_std[m][best_index])
            if not opt.real_world:
                greedy_choice[m]['(GUAPO) add error'] = str(all_add[m][best_index]) 
        else:
            if all_score[m][best_index_score[m]] > 0:
                greedy_choice[m]['(GUAPO) best frame'] = Names[best_index_score[m]]
                greedy_choice[m]['(GUAPO) best std'] = 'No GUAPO result, using score'
                if not opt.real_world:
                    greedy_choice[m]['(GUAPO) add error'] = str(all_add[m][best_index_score[m]])             

    # the oracle choice (directly choosing the image with the smallest ADD error)
    if not opt.real_world:
        for m in all_add:
            best_index = 0
            for index in range(len(all_add[m])):
                if all_add[m][index] < all_add[m][best_index]:
                    best_index = index
            print(m+' (oracle) best image is', Names[best_index])
            print(m+' (oracle) add error is', all_add[m][best_index])
            if all_score[m][best_index] > 0:
                greedy_choice[m]['(oracle) best frame'] = Names[best_index]
                greedy_choice[m]['(oracle) add error'] = str(all_add[m][best_index])

    # for ensemble (only support two or three networks)
    if opt.ensemble:
        visii.initialize_headless()
        folder = '../content/models/grocery/'
        obj1 = create_obj(
            name = opt.object + '_1',
            path_obj = folder + opt.object + "/google_16k/textured_simple.obj"
        )
        obj2 = create_obj(
            name = opt.object + '_2',
            path_obj = folder + opt.object + "/google_16k/textured_simple.obj"
        )

        if len(opt.model) == 3:
            M1 = [opt.model[0], opt.model[0], opt.model[1]]
            M2 = [opt.model[1], opt.model[2], opt.model[2]]
        else:
            M1 = [opt.model[0]]
            M2 = [opt.model[1]]

        if len(opt.model) == 3:
            m1 = opt.model[0]
            m2 = opt.model[1]
            m3 = opt.model[2]
            temp = all_score[m1] * all_score[m2] * all_score[m3]
            ensemble_index = temp > 0
            ensemble_count_3 = sum(ensemble_index)
            if ensemble_count_3 > 0:
                all_location_m1 = []
                all_location_m2 = []
                all_location_m3 = []

                all_orientation_m1 = []
                all_orientation_m2 = []
                all_orientation_m3 = []            
                
                Names_m = []

                for i in range(len(ensemble_index)):
                    if ensemble_index[i]:
                        all_location_m1.append(all_location[m1][i])
                        all_location_m2.append(all_location[m2][i])
                        all_location_m3.append(all_location[m3][i])
                        all_orientation_m1.append(all_orientation[m1][i])
                        all_orientation_m2.append(all_orientation[m2][i])
                        all_orientation_m3.append(all_orientation[m3][i])
                        Names_m.append(Names[i])

                all_disagreement_add = []
                if not opt.real_world:
                    all_add_m1 = all_add[m1][ensemble_index]
                    all_add_m2 = all_add[m2][ensemble_index]
                    all_add_m3 = all_add[m3][ensemble_index]

                for i in range(len(all_location_m1)):
                    # ADD
                    pos1 = all_location_m1[i]
                    quat1 = all_orientation_m1[i]
                    pos2 = all_location_m2[i]
                    quat2 = all_orientation_m2[i]
                    pos3 = all_location_m3[i]
                    quat3 = all_orientation_m3[i]                

                    d1 = get_add(obj1, obj2, pos1, pos2, quat1, quat2)
                    d2 = get_add(obj1, obj2, pos1, pos3, quat1, quat3)
                    d3 = get_add(obj1, obj2, pos2, pos3, quat2, quat3)

                    all_disagreement_add.append((d1+d2+d3) / 3.0)

                if not opt.real_world:
                    my_dpi = 100
                    fig, axs = plt.subplots(2, 3, figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)
                    
                    axs[0,0].scatter(all_disagreement_add, all_add_m1)
                    axs[0,0].set_xlabel('add disagreement')
                    axs[0,0].set_ylabel('add (model 1)')
                    temp_p = np.corrcoef(all_disagreement_add, all_add_m1)[0, 1]
                    temp_s = stats.spearmanr(all_disagreement_add, all_add_m1)[0]
                    corr_summary[m1]['all three'] = str(temp_p)
                    axs[0,0].set_title('P: ' + str(np.round(temp_p, 2)) + ' S: ' + str(np.round(temp_s, 2)))

                    axs[0,1].scatter(all_disagreement_add, all_add_m2)
                    axs[0,1].set_xlabel('add disagreement')
                    axs[0,1].set_ylabel('add (model 2)')
                    temp_p = np.corrcoef(all_disagreement_add, all_add_m2)[0, 1]
                    temp_s = stats.spearmanr(all_disagreement_add, all_add_m2)[0]
                    corr_summary[m2]['all three'] = str(temp_p)
                    axs[0,1].set_title('P: ' + str(np.round(temp_p, 2)) + ' S: ' + str(np.round(temp_s, 2)))

                    axs[0,2].scatter(all_disagreement_add, all_add_m3)
                    axs[0,2].set_xlabel('add disagreement')
                    axs[0,2].set_ylabel('add (model 3)')
                    temp_p = np.corrcoef(all_disagreement_add, all_add_m3)[0, 1]
                    temp_s = stats.spearmanr(all_disagreement_add, all_add_m3)[0]
                    corr_summary[m3]['all three'] = str(temp_p)
                    axs[0,2].set_title('P: ' + str(np.round(temp_p, 2)) + ' S: ' + str(np.round(temp_s, 2)))

                    fil = np.array(all_disagreement_add) <= 0.35
                    x = np.array(all_disagreement_add)[fil]
                    x_eval = np.linspace(0, max(x), 200)
                    index = 0
                    for data in [all_add_m1, all_add_m2, all_add_m3]:
                        y = np.array(data)[fil]
                        y_pred, score, label_nom, y_pred_up, label_up, y_pred_up_nointer, label_up_nointer = LR(x, y, x_eval)
                        axs[1,index].plot(x, y, 'b.', markersize=5, label='data')
                        axs[1,index].plot(x_eval, y_pred, 'b-', label=label_nom+' (nominal)')
                        axs[1,index].plot(x_eval, y_pred_up, 'r', linestyle='--', label=label_up+' (cover 90% data)')
                        axs[1,index].plot(x_eval, y_pred_up_nointer, 'r', linestyle='dotted', label=label_up_nointer+' (cover 90% data)')
                        axs[1,index].legend()
                        axs[1,index].set_xlabel('add disagreement')
                        axs[1,index].set_ylabel('add (model '+str(index+1)+')')
                        #axs[1,index].set_title('GP score: '+str(np.round(score, 2)))
                        index += 1

                    plt.tight_layout() 
                    plt.savefig(opt.resultf+'/result_ensemble_3'+'.png', dpi=my_dpi)
                    plt.close()      

                best_index = 0
                for index in range(len(all_disagreement_add)):
                    if all_disagreement_add[index] < all_disagreement_add[best_index]:
                        best_index = index
                print('(3-ensemble) best image is', Names_m[best_index])
                print('(3-ensemble) best disagreement is', all_disagreement_add[best_index])
                if not opt.real_world:
                    print('m1 (3-ensemble) add error is', all_add_m1[best_index])
                    print('m2 (3-ensemble) add error is', all_add_m2[best_index])
                    print('m3 (3-ensemble) add error is', all_add_m3[best_index])
                greedy_choice[m1]['(3-ensemble) best frame'] = Names_m[best_index]
                greedy_choice[m1]['(3-ensemble) best disagreement'] = str(all_disagreement_add[best_index])
                greedy_choice[m2]['(3-ensemble) best frame'] = Names_m[best_index]
                greedy_choice[m2]['(3-ensemble) best disagreement'] = str(all_disagreement_add[best_index])
                greedy_choice[m3]['(3-ensemble) best frame'] = Names_m[best_index]
                greedy_choice[m3]['(3-ensemble) best disagreement'] = str(all_disagreement_add[best_index])
                if not opt.real_world:
                    greedy_choice[m1]['(3-ensemble) add error'] = str(all_add_m1[best_index])
                    greedy_choice[m2]['(3-ensemble) add error'] = str(all_add_m2[best_index])
                    greedy_choice[m3]['(3-ensemble) add error'] = str(all_add_m3[best_index])            
            else:
                if all_score[m1][best_index_score[m1]] > 0:
                    greedy_choice[m1]['(3-ensemble) best frame'] = Names[best_index_score[m1]]
                    greedy_choice[m1]['(3-ensemble) best disagreement'] = 'No ensemble result, using score'
                    if not opt.real_world:
                        greedy_choice[m1]['(3-ensemble) add error'] = str(all_add[m1][best_index_score[m1]])
                if all_score[m2][best_index_score[m2]] > 0:
                    greedy_choice[m2]['(3-ensemble) best frame'] = Names[best_index_score[m2]]
                    greedy_choice[m2]['(3-ensemble) best disagreement'] = 'No ensemble result, using score'
                    if not opt.real_world:
                        greedy_choice[m2]['(3-ensemble) add error'] = str(all_add[m2][best_index_score[m2]])
                if all_score[m3][best_index_score[m3]] > 0:
                    greedy_choice[m3]['(3-ensemble) best frame'] = Names[best_index_score[m2]]
                    greedy_choice[m3]['(3-ensemble) best disagreement'] = 'No ensemble result, using score'
                    if not opt.real_world:
                        greedy_choice[m3]['(3-ensemble) add error'] = str(all_add[m3][best_index_score[m3]])
                if not opt.real_world:
                    corr_summary[m1]['all three'] = 'no result'
                    corr_summary[m2]['all three'] = 'no result'
                    corr_summary[m3]['all three'] = 'no result'


        ensemble_count_2 = {}
        for ii in range(len(M1)):
            m1 = M1[ii] 
            m2 = M2[ii]
            temp = all_score[m1] * all_score[m2]
            ensemble_index = temp > 0
            ensemble_count_2[m1+', '+m2] = sum(ensemble_index)
            if sum(ensemble_index) > 0:
                all_score_m1 = all_score[m1][ensemble_index]
                all_score_m2 = all_score[m2][ensemble_index]

                all_location_m1 = []
                all_location_m2 = []
                all_orientation_m1 = []
                all_orientation_m2 = []
                Names_m = []
                for i in range(len(ensemble_index)):
                    if ensemble_index[i]:
                        all_location_m1.append(all_location[m1][i])
                        all_location_m2.append(all_location[m2][i])
                        all_orientation_m1.append(all_orientation[m1][i])
                        all_orientation_m2.append(all_orientation[m2][i])
                        Names_m.append(Names[i])

                all_disagreement_add = []
                if not opt.real_world:
                    all_add_m1 = all_add[m1][ensemble_index]
                    all_add_m2 = all_add[m2][ensemble_index]

                for i in range(len(all_location_m1)):
                    # ADD
                    pos1 = all_location_m1[i]
                    quat1 = all_orientation_m1[i]
                    pos2 = all_location_m2[i]
                    quat2 = all_orientation_m2[i]
                    
                    all_disagreement_add.append(get_add(obj1, obj2, pos1, pos2, quat1, quat2))

                if not opt.real_world:
                    my_dpi = 100
                    fig, axs = plt.subplots(2, 2, figsize=(800/my_dpi, 600/my_dpi), dpi=my_dpi)
                                   
                    axs[0, 0].scatter(all_disagreement_add, all_add_m1)
                    axs[0, 0].set_xlabel('add disagreement')
                    axs[0, 0].set_ylabel('add (model 1)')
                    temp_p = np.corrcoef(all_disagreement_add, all_add_m1)[0, 1]
                    temp_s = stats.spearmanr(all_disagreement_add, all_add_m1)[0]
                    corr_summary[m1][m1+', '+m2] = str(temp_p)
                    axs[0, 0].set_title('P: ' + str(np.round(temp_p, 2)) + ' S: ' + str(np.round(temp_s, 2)))

                    axs[0, 1].scatter(all_disagreement_add, all_add_m2)
                    axs[0, 1].set_xlabel('add disagreement')
                    axs[0, 1].set_ylabel('add (model 2)')
                    temp_p = np.corrcoef(all_disagreement_add, all_add_m2)[0, 1]
                    temp_s = stats.spearmanr(all_disagreement_add, all_add_m2)[0]
                    corr_summary[m2][m1+', '+m2] = str(temp_p)
                    axs[0, 1].set_title('P: ' + str(np.round(temp_p, 2)) + ' S: ' + str(np.round(temp_s, 2)))

                    fil = np.array(all_disagreement_add) <= 0.35
                    if sum(fil) >= 1:
                        x = np.array(all_disagreement_add)[fil]
                    else:
                        x = np.array(all_disagreement_add)
                    x_eval = np.linspace(0, max(x), 200)
                    index = 0
                    for data in [all_add_m1, all_add_m2]:
                        y = np.array(data)[fil]
                        y_pred, score, label_nom, y_pred_up, label_up, y_pred_up_nointer, label_up_nointer = LR(x, y, x_eval)
                        axs[1,index].plot(x, y, 'b.', markersize=5, label='data')
                        axs[1,index].plot(x_eval, y_pred, 'b-', label=label_nom+' (nominal)')
                        axs[1,index].plot(x_eval, y_pred_up, 'r', linestyle='--', label=label_up+' (cover 90% data)')
                        axs[1,index].plot(x_eval, y_pred_up_nointer, 'r', linestyle='dotted', label=label_up_nointer+' (cover 90% data)')
                        axs[1,index].legend()
                        axs[1,index].set_xlabel('add disagreement')
                        axs[1,index].set_ylabel('add (model '+str(index+1)+')')
                        #axs[1,index].set_title('GP score: '+str(np.round(score, 2)))
                        index += 1

                    plt.tight_layout() 
                    plt.savefig(opt.resultf+'/result_ensemble_'+m1+m2+'.png', dpi=my_dpi)
                    plt.close()

                best_index = 0
                for index in range(len(all_disagreement_add)):
                    if all_disagreement_add[index] < all_disagreement_add[best_index]:
                        best_index = index
                print('(ensemble) best image is', Names_m[best_index])
                print('(ensemble) best disagreement is', all_disagreement_add[best_index])
                if not opt.real_world:
                    print('m1 (ensemble) add error is', all_add_m1[best_index])
                    print('m2 (ensemble) add error is', all_add_m2[best_index])
                greedy_choice[m1]['(ensemble) best frame'] = Names_m[best_index]
                greedy_choice[m1]['(ensemble) best disagreement'] = str(all_disagreement_add[best_index])
                greedy_choice[m2]['(ensemble) best frame'] = Names_m[best_index]
                greedy_choice[m2]['(ensemble) best disagreement'] = str(all_disagreement_add[best_index])
                if not opt.real_world:
                    greedy_choice[m1]['(ensemble) add error'] = str(all_add_m1[best_index])
                    greedy_choice[m2]['(ensemble) add error'] = str(all_add_m2[best_index])
            else:
                if all_score[m1][best_index_score[m1]] > 0:
                    greedy_choice[m1]['(ensemble) best frame'] = Names[best_index_score[m1]]
                    greedy_choice[m1]['(ensemble) best disagreement'] = 'No ensemble result, using score'
                    if not opt.real_world:
                        greedy_choice[m1]['(ensemble) add error'] = str(all_add[m1][best_index_score[m1]])
                if all_score[m2][best_index_score[m2]] > 0:
                    greedy_choice[m2]['(ensemble) best frame'] = Names[best_index_score[m2]]
                    greedy_choice[m2]['(ensemble) best disagreement'] = 'No ensemble result, using score'
                    if not opt.real_world:
                        greedy_choice[m2]['(ensemble) add error'] = str(all_add[m2][best_index_score[m2]])
                if not opt.real_world:
                    corr_summary[m1][m1+', '+m2] = 'no result'
                    corr_summary[m2][m1+', '+m2] = 'no result'

        visii.cleanup()

    # Save greedy_choice
    with open(opt.resultf+'/greedy_choice.json', 'w') as fp:
        json.dump(greedy_choice, fp, indent=4)

    # Performance (AUC curve)
    if not opt.real_world:
        my_dpi = 100
        plt.figure(figsize=(400/my_dpi, 300/my_dpi), dpi=my_dpi)
        thres = np.linspace(0, 0.2, 50)
        label = []
        for m in opt.model:
            r = []
            for thre in thres:
                r.append(sum(all_add[m] <= thre)/len(all_add[m]))
            label.append(m)
            plt.plot(thres, r)
        plt.ylim([0, 1])
        plt.legend(label)
        plt.tight_layout() 
        plt.savefig(opt.resultf+'/result_performance.png', dpi=my_dpi)
        plt.close()

    # only keep the detected frames
    if not opt.real_world:
        total_number = {}
        detected_number = {}
        ratio = {}
        for m in all_score:
            total_number[m] = len(all_score[m])
            detected_index = all_score[m] > 0
            detected_number[m] = sum(detected_index)
            ratio[m] = detected_number[m]/total_number[m]
            all_score[m] = all_score[m][detected_index]
            all_pose_std[m] = all_pose_std[m][detected_index]
            all_add[m] = all_add[m][detected_index]

    if not opt.real_world:
        my_dpi = 100
        fig, axs = plt.subplots(1, 2, figsize=(800/my_dpi, 300/my_dpi), dpi=my_dpi)
        label = []
        corrcoef = [[], []]
        spearmanr = [[], []]
        for m in all_score:
            label.append(m)
            axs[0].scatter(-all_score[m], all_add[m])
            axs[1].scatter(all_pose_std[m], all_add[m])

        axs[0].legend(label)
        axs[1].set_xlim([0, 0.5])

        for m in all_score:
            corrcoef[0].append(np.corrcoef(-all_score[m], all_add[m])[0, 1])
            corrcoef[1].append(np.corrcoef(all_pose_std[m], all_add[m])[0, 1])
            spearmanr[0].append(stats.spearmanr(-all_score[m], all_add[m])[0])
            spearmanr[1].append(stats.spearmanr(all_pose_std[m], all_add[m])[0])

            corr_summary[m]['score'] = str(stats.spearmanr(-all_score[m], all_add[m])[0])
            corr_summary[m]['GUAPO (loc)'] = str(stats.spearmanr(all_pose_std[m], all_add[m])[0])

        axs[0].set_xlabel('-1*centroid score')
        axs[0].set_ylabel('add')
        axs[0].set_title('P: ' + str(np.round(corrcoef[0], 2)) + '\n S: ' + str(np.round(spearmanr[0], 2)))        
        axs[1].set_xlabel('location std (GUAPO)')
        axs[1].set_ylabel('add')
        axs[1].set_title('P: ' + str(np.round(corrcoef[1], 2)) + '\n S: ' + str(np.round(spearmanr[1], 2)))    
        plt.tight_layout() 
        plt.savefig(opt.resultf+'/result_correlation.png', dpi=my_dpi)
        plt.close()

    if not opt.real_world:
        print('Total number: ', total_number)
        print('Detected number: ', detected_number)
        print('Detected/Total: ', ratio)
        if opt.ensemble:
            print('ensemble number (two models): ', ensemble_count_2)
            if len(opt.model) == 3:
                print('ensemble number (three models): ', ensemble_count_3)

        corr_summary['Total number'] = str(total_number[opt.object])
        
        for temp in detected_number:
            detected_number[temp] = str(detected_number[temp])
        corr_summary['Detected number'] = detected_number
        
        for temp in ratio:
            ratio[temp] = str(ratio[temp])
        corr_summary['Detected/Total'] = ratio
        
        for temp in ensemble_count_2:
            ensemble_count_2[temp] = str(ensemble_count_2[temp])
        corr_summary['Ensemble number (two)'] = ensemble_count_2
        
        if len(opt.model) == 3:
            corr_summary['Ensemble model (three)'] = str(ensemble_count_3)

        # Save correlation result
        with open(opt.resultf+'/correlation_summary.json', 'w') as fp:
            json.dump(corr_summary, fp, indent=4)
