from torch.nn import Conv2d, Dropout
import math
import os.path as osp
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
_tokenizer = _Tokenizer()
from torchvision.models import resnet50

# import sys
# sys.path.append("/l/users/sanoojan.baliah/Felix/LaFTer")
from utils.model_utils import *
from utils.utils import *
_tokenizer = _Tokenizer()
from functools import reduce
from operator import mul
from utils.data_utils import ds_specific_templates
from .svl import EmbModel, AdapterMLP

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')
    # model_path="all_weights/RemoteCLIP-ViT-B-32.pt"
    # model_path ="all_weights/RS5M_ViT-B-32.pt"1
    print("_______________________________")
    print(model_path)
    print("_______________________________")
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    
    # Load model


    return model

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

def process_json_files(imagenet_file, dataset_file, output_file, dataset, desc_noise):
    with open(imagenet_file, 'r') as file:
        original_data = json.load(file)

    new_data = {"other": [value for values in original_data.values() for value in values]}

    if dataset=="EuroSAT":
        samples=int(250*desc_noise)
    elif dataset=="Resisc45":
        samples=int(45*50*desc_noise)
    elif dataset=="AID":
        samples=int(30*50*desc_noise)
    elif dataset=="PatternNet":
        samples=int(38*50*desc_noise)
    else:
        # Input samples for other datasets
        samples=int(input("Enter the number of noise samples for the other datasets: "))
    print("Number of noise samples: ", samples)

    sampled_values = random.sample(new_data['other'], samples)
    sampled_data = {"other": sampled_values}

    with open(dataset_file, 'r') as file:
        dataset_data = json.load(file)

    #combine the two dictionaries
    dataset_data.update(sampled_data)
        
    # Save the updated dataset_data to the output file
    with open(output_file, 'w') as file:
        json.dump(dataset_data, file, indent=4)


class LaFTerUFT(nn.Module):

    def __init__(self, model, classes, templates, ds_templates=None, device='cuda', log=None, dataset_name=None, txt_cls=None, cfg=None):
        super(LaFTerUFT, self).__init__()
        self.adapter_pl = None
        self.image_features_frozen = None
        self.device = device
        self.cfg = cfg
        self.dataset_templates = ds_templates
        self.classes = classes
        self.dataset_name = dataset_name
        self.classes = classes
        self.txt_cls = txt_cls
        self.log = log
        patch_size = (16, 16)
        self.model = model.to(device)
        self.templates = templates
        self.backbone_out_size = 512
        self.hidden_size = 768 # todo for ViT-L use 1024 - for ViT-B use 768
        self.num_tokens = 50 # todo all experiments are run with num_tokens = 50
        self.prompt_proj = nn.Identity()
        self.hidden_size_text = 512
        self.num_tokens_text = 77
        prompt_dim = self.hidden_size
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        self.prompt_dropout = Dropout(0.0)
        self.adapter = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False)).to(device)
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_tokens, self.hidden_size), requires_grad=True)
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()
        self.text_features = self.txt_features()
        self.class_desc_emb = self.gen_emb()
        self.svl_enc = None
        self.svl_adapter = None
        
    def train_txt_clas(self, criteria):
        noise_std = 0.1
        noise = torch.randn(self.txt_features_for_text_cls.shape) * noise_std
        txt_feas = self.txt_features_for_text_cls
        txt_label = self.labels_for_text_cls
        feas = (self.adapter(txt_feas.to(torch.float32) + noise.cuda()))
        loss = criteria(feas, txt_label)
        return loss

    def txt_features_for_text_cls(self):

        if self.txt_cls== 'cls_only':
            gpt3_prompts = None
            desc, labels_for_descriptions = gen_labels_with_classes(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == 'templates_only':
            gpt3_prompts = self.templates
            desc, labels_for_descriptions = gen_labels_with_templates(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == 'lafter':
            # generic prompts + templates

            if self.dataset_name not in lafter_datasets:
                raise ValueError('Invalid dataset name for LaFTer')
            
            # process_json_files('./descriptions/generic/ImageNet.json', f'./descriptions/generic/{self.dataset_name}.json', f'./descriptions/generic/{self.dataset_name}_noised.json', self.dataset_name)

            if self.cfg.desc_noise>0:
                process_json_files('./descriptions/generic/ImageNet.json', f'./descriptions/generic/{self.dataset_name}.json', f'./descriptions/generic/{self.dataset_name}_noised.json', self.dataset_name, self.cfg.desc_noise)
                path_to_file = f'./descriptions/generic/{self.dataset_name}_noised.json'
            else:
                path_to_file = f'./descriptions/generic/{self.dataset_name}.json'


            with open(path_to_file) as f:
                gpt3_prompts = json.load(f)

            desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)

            templates, labels_for_templates = gen_labels_with_templates(self.classes, descriptions=self.dataset_templates)

            desc += templates
            labels_for_descriptions += labels_for_templates

        elif self.txt_cls == 'zero_shot':
            pass

        else:
            raise ValueError('Invalid txt_cls argument')


        if self.txt_cls in ['cls_only', 'templates_only', 'lafter']:

            Path(f'embeddings').mkdir(parents=True, exist_ok=True)

            if os.path.isfile(f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt'):
                zeroshot_weights = torch.load(f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt')
                print('******** Loaded Already Saved Embeddings *********')
                labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

            else:
                print('******** No Embeddings Found --- Saving New Embeddings *********')

                labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

                zeroshot_weights = []
                with torch.no_grad():
                    for classname in tqdm(desc):
                        text = tokenize(classname).cuda()  # tokenize # (50, 77) --> 50 templates/texts from GPT
                        class_embeddings = self.model.encode_text(
                            text)  # embed with text encoder # (50, 512) --> embeddings for all 50 texts
                        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # L2 norm of the embeddings (dim 2)
                        zeroshot_weights.append(class_embeddings)
                    zeroshot_weights = torch.stack(zeroshot_weights).cuda()  # (512, 10) --> 512 embeddings for 10 classes'
                    # torch.save(zeroshot_weights, f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt')

            return zeroshot_weights.squeeze(), labels_for_descriptions

        else:
            return None, None
        
    def gen_emb(self):

        # Map unique string values to numerical indices
        unique_groups, group_indices = torch.unique(self.labels_for_text_cls, return_inverse=True)

        # Get the total number of unique groups
        total_groups = len(unique_groups)

        # Ensure the number of specified groups is valid
        if len(self.classes) > total_groups:
            raise ValueError("The specified number of groups is greater than the total number of unique groups.")

        # Split the indices into the specified number of groups
        group_indices_split = torch.chunk(torch.arange(total_groups), len(self.classes))

        # Create a list to store tensors for each group
        group_list = []

        # Iterate over each group and create tensors for the "other" tensor
        for indices in group_indices_split:
            subset_other_tensor = self.txt_features_for_text_cls[group_indices == indices[0]]  # Assuming group indices are consistent
            group_list.append(subset_other_tensor)

        # Stack the tensors to create a tensor of shape (num_groups, group_dim, emb_dim)
        other_tensor_split = torch.stack(group_list).mean(axis=1)

        return other_tensor_split.T

    def txt_features(self):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classes):
                texts = [template.format(classname) for template in self.templates]  # format with class
                texts = tokenize(texts).cuda()  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def image_features(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def eval_clip(self, x):
        with torch.no_grad():
            img_features_2 = self.incorporate_prompt(x)
            img_features_2 = self.embeddings_after_prompts(img_features_2)
            img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

    def forward(self, x):
        # only used for 0-shot-eval
        with torch.no_grad():
            img_features = self.image_features(x)
            pseudo_label = img_features @ self.text_features
        return pseudo_label

    def forward_pl_zeroshot(self, x):
        with torch.no_grad():
            if self.cfg.ve_unshared:
                print('******** Unsharing Vision Encoders *********')
                img_features = self.image_features_frozen_pl(x)
            else:
                img_features = self.image_features(x)

            if self.cfg.desc_emb:
                pseudo_label = img_features @ self.class_desc_emb
            else:
                pseudo_label = img_features @ self.text_features.float()
        return pseudo_label

    def forward_aug_with_prompts(self, x2):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        img_features_2 = self.incorporate_prompt(x2)
        img_features_2 = self.embeddings_after_prompts(img_features_2)
        img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

    def txt_cls_init(self):
        import copy
        self.adapter_pl = copy.deepcopy(self.adapter)
        if self.cfg.classifer_random_weights:
            print('******** Initializing Classifier with Random Weights *********')
            self.adapter_pl.apply(weights_init)

        if self.cfg.ve_unshared:
            self.image_features_frozen = copy.deepcopy(self.model.visual)

    def svl_adapter_init(self, args):

        self.svl_enc = EmbModel(base_encoder=resnet50, args={'pretrained': True, 'projection_dim': 128, 'num_train': 1000, 'device': 'cuda', 'store_embeddings': False}).to(self.device)
        self.svl_adapter = AdapterMLP(num_classes=len(self.classes), input_size=2048, hidden_size=256).to(self.device)
        # breakpoint()
        svl_enc_apth = args.svl_model_path + '/eurosat_resnet50_simclr_2024_01_07__17_03_13.pt'
        svl_adapter_path = args.svl_model_path + '/adapter.pt'
        # breakpoint()
        checkpoint_enc = torch.load(svl_enc_apth)
        checkpoint_adapter = torch.load(svl_adapter_path)

        self.svl_enc.load_state_dict(checkpoint_enc['state_dict'])
        self.svl_adapter.load_state_dict(checkpoint_adapter) 

        # freeze svl_enc and adapter
        for param in self.svl_enc.parameters():
            param.requires_grad = False

        for param in self.svl_adapter.parameters():
            param.requires_grad = False 

    def forward_svl(self, x):
        # with torch.no_grad():
        op = self.svl_enc(x, only_feats=True)
        op = self.svl_adapter(op['feat'])
        return op

    def image_features_frozen_pl(self, images):
        with torch.no_grad():
            image_features = self.image_features_frozen(images.type(self.model.dtype))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def forward_normal_for_pl(self, x1):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        with torch.no_grad():
            if self.cfg.ve_unshared:
                 img_features_1 = self.image_features_frozen_pl(x1)   
            else:
                img_features_1 = self.image_features(x1)
            pseudo_label = self.adapter_pl(img_features_1.float()).detach()
        return pseudo_label

    def incorporate_prompt(self, x, teacher=False):
        B = x.shape[0]
        x = self.patch_embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x

    def patch_embeddings(self, x: torch.tensor):
        return self.model.visual.embeddings_patch(x)

    def embeddings_after_prompts(self, x: torch.tensor):
        return self.model.visual.forward_after_patch_embeddings(x)

    def positional_embeddings_for_text(self, x: torch.tensor):
        return self.model.positional_embeddings(x)

    def embeddings_after_prompts_for_text(self, x: torch.tensor):
        return self.model.embeddings_after_prompting(x)


@TRAINER_REGISTRY.register()
class LaFTer(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")
        self.model = LaFTerUFT(model=clip_model, classes=classnames,
                                          templates=['a photo of a {}'], ds_templates = ds_specific_templates[cfg.DATASET.NAME], dataset_name= cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg)
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        #  freeze clip
        for param in self.model.model.parameters():
            param.requires_grad = False

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, custom_tfm_test=te_transform, custom_tfm_train=tr_transforms)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def parse_batch_train(self, batch):

        if isinstance(batch, list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device)
        else:
            input = batch['img']
            input = input.to(self.device)

        label = batch["label"]
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
