from email.mime import image
from re import U
import model_data_maneger
import os

import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import hydra
# from omegaconf import DictConfig
from simswap.src.simswap import SimSwap
import cv2
import numpy as np

def get_perturbation(data_loader, pert_gen_net, config):
    return torch.load(config.global_settings.universal_perturbation_path)


def load_demo():
    pass

def vaild(config):
    data_loader, stargan, pert_gen_net, simswap= model_data_maneger.prepare(config)
    # model_data_maneger.getconfig("config.yaml")

    #att_gan
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0

    perturbs = get_perturbation(data_loader, pert_gen_net, config)

    tf = transforms.Compose([
            # transforms.CenterCrop(170),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    image = cv2.cvtColor(cv2.imread(str(config.global_settings.demo_images)), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # images = simswap.run_detect_align(image=image, for_id=False)
    
    image = tf(image).unsqueeze(0)
    image = (image.cuda() + perturbs).cpu()
    image = transforms.Resize([1080, 1920])(image)
    out_file = os.path.join(config.global_settings.demo_result + '/stargan_per.jpg')
    vutils.save_image(image.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.))  
    
    image = np.array(transforms.ToPILImage()(image.squeeze(0)))
    print(type(image))
    print(image.shape)
    images, att_transforms, _ = simswap.run_detect_align(image=image, for_id=False, crop_size=256)

    images = torch.stack(
            [tf(x) for x in images], dim=0
        )
    for idx, (img, att_a, c_org) in enumerate(data_loader):
        img = img.cuda() if config.global_settings.gpu else img
        img = images.cuda() if config.global_settings.gpu else images
        c_org = c_org.cuda() if config.global_settings.gpu else c_org
        c_org = c_org.type(torch.float)

        #get perturbation and gen image
       
        x_noattack_list, x_attack_list = stargan.test_universal_model_level(img, c_org, perturbs, config.stargan)

        x_total_list = [img, img + perturbs]
        for j in range(len(x_attack_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_attack_list[j]

            x_total_list.append(x_noattack_list[j])
            x_total_list.append(x_attack_list[j])

            l1_error += F.l1_loss(gen, gen_noattack)
            l2_error += F.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if F.mse_loss(gen, gen_noattack) > 0.05:
                n_dist += 1
            n_samples += 1
        
        # # save origin image
        # out_file = config.global_settings.result_path + '/stargan_original.jpg'
        # vutils.save_image(imgs.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.)) 
        x_concat = torch.cat(x_total_list, dim=3)
        # out_file = os.path.join(config.global_settings.result_path + '/stargan_gen_{}.jpg'.format(idx))
        # out_file = os.path.join(config.global_settings.result_path + '_per/stargan_per_{}.jpg'.format(idx))
        out_file = os.path.join(config.global_settings.demo_result + '/stargan_per_{}.jpg'.format(idx))
        vutils.save_image(x_concat, out_file, nrow=1, normalize=True, range=(-1., 1.))
        
        #gen deepfake image and adversarial example
        # for j in range(len(x_attack_list)):
        #     #save deepfake image
        #     gen_noattack = x_noattack_list[j]
        #     out_file = config.global_settings.result_path + '/stargan_gen_{}.jpg'.format(j)
        #     vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, range=(-1., 1.))

        #     # save deepfake adversarial example
        #     gen = x_fake_list[j]
        #     out_file = config.global_settings.result_path + '/stargan_advgen_{}.jpg'.format(j)
        #     vutils.save_image(gen, out_file, nrow=1, normalize=True, range=(-1., 1.))

        break
    print('stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
    

def save_image(iamges, result_path):
    pass


def trian():
    data_loader, config, stargan, pert_gen_net = model_data_maneger.prepare()
    # model_data_maneger.getconfig("config.yaml")

    #att_gan
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (imgs, att_a, c_org) in enumerate(data_loader):
        imgs = imgs.cuda() if config.global_settings.gpu else imgs
        c_org = c_org.cuda() if config.global_settings.gpu else att_a
        c_org = c_org.type(torch.float)

        #get perturbation and gen image
        perturbs = get_perturbation(data_loader, pert_gen_net, config)
        x_noattack_list, x_attack_list = stargan.test_universal_model_level(imgs, c_org, perturbs, config.stargan)

        
        for j in range(len(x_attack_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_attack_list[j]

            l1_error += F.l1_loss(gen, gen_noattack)
            l2_error += F.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if F.mse_loss(gen, gen_noattack) > 0.05:
                n_dist += 1
            n_samples += 1
        
        # # save origin image
        # out_file = config.global_settings.result_path + '/stargan_original.jpg'
        # vutils.save_image(imgs.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.)) 

        
        #gen deepfake image and adversarial example
        # for j in range(len(x_attack_list)):
        #     #save deepfake image
        #     gen_noattack = x_noattack_list[j]
        #     out_file = config.global_settings.result_path + '/stargan_gen_{}.jpg'.format(5* idx + j)
        #     vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, range=(-1., 1.))

        #     # save deepfake adversarial example
        #     gen = x_fake_list[j]
        #     out_file = config.global_settings.result_path + '/stargan_advgen_{}.jpg'.format(5 * idx + j)
        #     vutils.save_image(gen, out_file, nrow=1, normalize=True, range=(-1., 1.))

        # if idx == 50:
        #     break
    print('stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))



def test():
    pass
@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    vaild(config)

if __name__ == "__main__" :
    main()