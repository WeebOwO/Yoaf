import os
import gif
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import data_config

def load(root_data_path, test_id, post):
    return np.load(os.path.join(root_data_path, test_id + post), allow_pickle=True)
    
fold_cnt = 0
root_data_path = data_config['data_dir']
output_dir = f"error/{fold_cnt}"
fp_th = 5

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def check(label, pbb):
    p, cz, cy, cx, d = pbb
    
    for ground_truth in label:
        tz, ty, tx, d = ground_truth
        dist_squ = pow(tz - cz, 2) + pow(ty - cy, 2) + pow(tx - cx, 2)

        if dist_squ < d * d:
            return False
    
    return p > 0.3


def output_fp():
    pbb_data_dir = f'experiment/fold{fold_cnt}/res'
    pbb_data_list = os.listdir(pbb_data_dir)
   
    id_list = [id[:-8] for id in pbb_data_list]

    for id in id_list:
        if len(id) < 3:
            continue
        
        print(f"Current process id is {id}")
        pbbs = load(pbb_data_dir, id, "_pbb.npy")
        ground_truth = load(root_data_path, id, "_bboxes.npy")
        img = load(root_data_path, id, ".npy")
        fp_iter = 0
        t_tier = 0
        tp_iter = 0
        
        # for box in ground_truth:
        #     fig, ax = plt.subplots()
        #     ax.set_axis_off()
        #     cz, cy, cx, d = box
        #     slice_data = img[0][int(cz)]
        #     plt.imshow(slice_data, cmap='gray')
        #     rect = patches.Rectangle((cx - d, cy - d), d * 2, d * 2, linewidth=1, edgecolor='red' ,facecolor='none')
        #     ax.add_patch(rect)
        #     ax.text(cx + 5, cy + 5, f"{1:.3f}", color='red')
        #     plt.savefig(os.path.join(output_dir, f"{id}_gt_{t_tier}.png"), bbox_inches=0)
        #     t_tier += 1
        #     plt.close(fig)
       
        _, depth, height, width = img.shape
        
        for pbb in pbbs:
            p, cz, cy, cx, d = pbb
            if cz >= depth or cz < 0: 
                continue
            if fp_iter > fp_th:
                break
            isHFP = check(ground_truth, pbb)
            slice_data_top = img[0][int(max(cz - 1, 0))]
            slice_data_mid = img[0][int(cz)]
            slice_data_bot = img[0][int(min(cz + 1, depth))]
            p, cz, cy, cx, d = pbb
            
            if isHFP:
                continue
                frames = []
                fig, ax = plt.subplots()
                ax.set_axis_off()
                
                @gif.frame
                def frame1():
                # show top
                    plt.imshow(slice_data_top, cmap='gray')
                    rect = patches.Rectangle((cx - d, cy - d), d * 2, d * 2, linewidth=1, edgecolor='red' ,facecolor='none')
                    ax.add_patch(rect)
                    ax.text(cx + 5, cy + 5, f"{p:.3f}", color='red')
                    
                @gif.frame
                def frame2():
                    # show mid   
                    ax.set_axis_off()
                    plt.imshow(slice_data_mid, cmap='gray')
                    rect = patches.Rectangle((cx - d, cy - d), d * 2, d * 2, linewidth=1, edgecolor='red' ,facecolor='none')
                    ax.add_patch(rect)
                    ax.text(cx + 5, cy + 5, f"{p:.3f}", color='red')
                  
                # show bottom
                @gif.frame
                def frame3():
                    ax.set_axis_off()
                    plt.imshow(slice_data_bot, cmap='gray')
                    rect = patches.Rectangle((cx - d, cy - d), d * 2, d * 2, linewidth=1, edgecolor='red' ,facecolor='none')
                    ax.add_patch(rect)
                    ax.text(cx + 5, cy + 5, f"{p:.3f}", color='red')
                    
                frames.append(frame1())
                frames.append(frame2())
                frames.append(frame3())
                
                gif.save(frames, os.path.join(output_dir, f"{id}_fp_{fp_iter}.gif"), duration=10)
                fp_iter += 1
                
                plt.close(fig)
                
            else:
                if tp_iter > 2:
                    break
                fig, ax = plt.subplots()
                ax.set_axis_off()
                plt.imshow(slice_data_mid, cmap='gray')
                rect = patches.Rectangle((cx - d, cy - d), d * 2, d * 2, linewidth=1, edgecolor='red' ,facecolor='none')
                ax.add_patch(rect)
                ax.text(cx + 5, cy + 5, f"{p:.3f}", color='green')
                plt.savefig(os.path.join(output_dir, f"{id}_tp_{tp_iter}.png"), bbox_inches=0)
                tp_iter += 1
                plt.close(fig)
            
    return 

if __name__ == "__main__":
    output_fp()