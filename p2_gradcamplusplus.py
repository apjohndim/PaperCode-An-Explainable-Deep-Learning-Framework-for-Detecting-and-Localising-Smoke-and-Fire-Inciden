


# https://keisen.github.io/tf-keras-vis-docs/






import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam
import os


#%%

# Create GradCAM++ object

def gradcamplusplus (items_no,predictions_all,labels,data,info,model3,verbose = False,show=False, save = True, base_path='C:\\Users\\User\\'):
    gradcam = GradcamPlusPlus(model3,
                              model_modifier=ReplaceToLinear(),
                              clone=True)
    
    def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
        return (output[0,0],output[0,1])
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])   
    
    

    n = 121
    for item in items_no:
        
        predicted_label = predictions_all[item]
        predicted_label_ling = 'PoFS' if predicted_label == 1 else 'No incident'
        true_label = labels[item,1]
        true_label_ling = 'PoFS' if true_label == 1 else 'No incident'
        no = ['a{}'.format(n),'v{}'.format(n)]
        n+=1     
        
        
        
        instance = data[item,:,:,:]
        the_image = instance
        instance = np.expand_dims(instance, axis=0)
        # Generate cam with GradCAM++
        cam = gradcam(score_function, instance)
        #score = CategoricalScore([0,1])
        #cam = gradcam(score, img_tensor)
    
        colored = the_image
        the_image =  rgb2gray(the_image)
        
        f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))


        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        ax1.imshow(the_image)
        ax1.imshow(heatmap, alpha=0.5)
        ax1.axis('off')
        
        ax2.imshow(colored)
        ax2.axis('off')
        
        
        ax1.set_title('Fused image',fontsize = 18)
        ax2.set_title('Original',fontsize = 18)
        f.suptitle('\n{} {}\n True Label: {} \n Predicted: {}'.format(no[0],no[0],true_label_ling,predicted_label_ling), fontsize=22)        
        
        
        
        
        
        plt.tight_layout()
        plt.show()
        
        
        name = '{}_pred{}_is{}'.format(no[0],predicted_label_ling,true_label_ling)
        
        
        
        if predicted_label == 1 and true_label == 1:
            save_path = os.path.join(base_path,'TP')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        if predicted_label == 1 and true_label == 0:
            save_path = os.path.join(base_path,'FP')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        if predicted_label == 0 and true_label == 1:
            save_path = os.path.join(base_path,'FN')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        if predicted_label == 0 and true_label == 0:
            save_path = os.path.join(base_path,'TN')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        
        
    
    
        f.savefig(os.path.join(save_path,name))
        plt.close()


#%%

def scorecam (items_no,predictions_all,labels,data,info,model3,verbose = False,show=False, save = True, base_path='C:\\Users\\User\\'):
    
    from tf_keras_vis.scorecam import Scorecam
    from tf_keras_vis.utils import num_of_gpus    
    
    
    scorecam = Scorecam(model3)
    
    def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
        return (output[0,0])
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])   
    
    

    
    for item in items_no:
        
        predicted_label = predictions_all[item]
        predicted_label_ling = 'Positive' if predicted_label == 1 else 'Negative'
        true_label = labels[item,1]
        true_label_ling = 'Positive' if true_label == 1 else 'Negative'
        no = info[item,:]       
        
        
        
        instance = data[item,:,:,:]
        the_image = instance
        instance = np.expand_dims(instance, axis=0)
        # Generate cam with GradCAM++
        score = CategoricalScore([predicted_label])
        cam = scorecam(score, instance, penultimate_layer=-1)
        #score = CategoricalScore([0,1])
        #cam = gradcam(score, img_tensor)
    
        colored = the_image
        the_image =  rgb2gray(the_image)
        
        f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))


        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        ax1.imshow(the_image)
        ax1.imshow(heatmap, alpha=0.4)
        ax1.axis('off')
        
        ax2.imshow(colored)
        ax2.axis('off')
        
        
        ax1.set_title('Fused image',fontsize = 18)
        ax2.set_title('Original',fontsize = 18)
        f.suptitle('\n{} {}\n True Label: {} \n Predicted: {}'.format(no[0],no[1],true_label_ling,predicted_label_ling), fontsize=22)        
        
        
        
        
        
        plt.tight_layout()
        plt.show()
        
        
        name = '{}_pred{}_is{}'.format(no[0],predicted_label_ling,true_label_ling)
        
        
        
        if predicted_label == 1 and true_label == 1:
            save_path = os.path.join(base_path,'TP')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        if predicted_label == 1 and true_label == 0:
            save_path = os.path.join(base_path,'FP')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        if predicted_label == 0 and true_label == 1:
            save_path = os.path.join(base_path,'FN')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        if predicted_label == 0 and true_label == 0:
            save_path = os.path.join(base_path,'TN')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        
        
    
    
        f.savefig(os.path.join(save_path,name))
        plt.close()



def gradcam (items_no,predictions_all,labels,data,info,model3,verbose = False,show=False, save = True, base_path='C:\\Users\\User\\'):
    
    from tf_keras_vis.scorecam import Scorecam
    from tf_keras_vis.utils import num_of_gpus    
    
    
    gradcam = Gradcam(model3)
    
    def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
        return (output[0,0])
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])   
    
    

    
    for item in items_no:
        
        predicted_label = predictions_all[item]
        predicted_label_ling = 'Positive' if predicted_label == 1 else 'Negative'
        true_label = labels[item,1]
        true_label_ling = 'Positive' if true_label == 1 else 'Negative'
        no = info[item,:]       
        
        
        
        instance = data[item,:,:,:]
        the_image = instance
        instance = np.expand_dims(instance, axis=0)
        # Generate cam with GradCAM++
        score = CategoricalScore([predicted_label])
        cam = gradcam(score, instance, penultimate_layer=-1)
        #score = CategoricalScore([0,1])
        #cam = gradcam(score, img_tensor)
    
        colored = the_image
        the_image =  rgb2gray(the_image)
        
        f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))


        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        ax1.imshow(the_image)
        ax1.imshow(heatmap, alpha=0.6)
        ax1.axis('off')
        
        ax2.imshow(colored)
        ax2.axis('off')
        
        
        ax1.set_title('Fused image',fontsize = 18)
        ax2.set_title('Original',fontsize = 18)
        f.suptitle('\n{} {}\n True Label: {} \n Predicted: {}'.format(no[0],no[1],true_label_ling,predicted_label_ling), fontsize=22)        
        
        
        
        
        
        plt.tight_layout()
        plt.show()
        
        
        name = '{}_pred{}_is{}'.format(no[0],predicted_label_ling,true_label_ling)
        
        
        
        if predicted_label == 1 and true_label == 1:
            save_path = os.path.join(base_path,'TP')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        if predicted_label == 1 and true_label == 0:
            save_path = os.path.join(base_path,'FP')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        if predicted_label == 0 and true_label == 1:
            save_path = os.path.join(base_path,'FN')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        if predicted_label == 0 and true_label == 0:
            save_path = os.path.join(base_path,'TN')
            if not os.path.exists (save_path) : os.mkdir(save_path)
        
        
    
    
        f.savefig(os.path.join(save_path,name))
        plt.close()

