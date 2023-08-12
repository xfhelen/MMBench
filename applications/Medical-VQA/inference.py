import torch 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import misc.io as io
from core.datasets import loaders_factory
from core.models import model_factory
import argparse
import yaml
# read config name from CLI argument --path_config
parser = argparse.ArgumentParser(
    description='Read config file name',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--path_config', default='applications/Medical-VQA/config/idrid_regions/single/default_consistency.yaml', type=str, help='path to a yaml options file') 
parser.add_argument('--need_data', default=1, type=int, help='choose whether to use real data') 
parser.add_argument('--options', default="normal", type=str, help='choose the model part') 
args = parser.parse_args()

def main():
    # read config file
    with open(args.path_config, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # define device as gpu (if available) or cpu
    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')
    
    if args.need_data == 1:
    # load data
        train_loader, vocab_words, vocab_answers, index_unk_answer = loaders_factory.get_vqa_loader('train', config, shuffle=True)
        val_loader = loaders_factory.get_vqa_loader('val', config) 
    # create model
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    # load weights for the loss function. The weights are provided in the dataset. See the script compute_answer_weights.py to check how they were computed
    if args.need_data == 1 and 'weighted_loss' in config:
        if config['weighted_loss']:
            answer_weights = io.read_weights(config) # if use of weights is required, read them from folder where they were previously saved using compute_answer_weights scripts
        else:
            answer_weights = None # If false, just set variable to None
    else:
        answer_weights = None
    options = args.options
    
    for epoch in range(0, config['epochs']+1):
        model.eval()
        with torch.no_grad():
            if options == 'encoder' or options == 'normal':
                for i, sample in enumerate(val_loader):
                    batch_size = sample['question'].size(0)

                    # move data to GPU
                    question = sample['question'].to(device)
                    visual = sample['visual'].to(device)
                    answer = sample['answer'].to(device)
                    question_indexes = sample['question_id'] # keep in cpu
                    if 'maskA' in sample:
                        mask_a = sample['maskA'].to(device)
                    flag = sample['flag'].to(device)

                    # get output
                    if 'maskA' in sample:
                        output = model(options, visual, question, mask_a)
                    else:
                        output = model(options, visual, question)
                    if i == 10:
                        break
            else : 
                for i in range(50):
                    output = model(options, 1,2,3)
                    if i == 50:
                        break
        break

    
    # pytorch profiler
    options == "normal"
    with torch.no_grad():
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './log/vqa_agg'),
            record_shapes=True,
            with_stack=True)
        with prof as p: 
            for i, sample in enumerate(val_loader):
                batch_size = sample['question'].size(0)

                # move data to GPU
                question = sample['question'].to(device)
                visual = sample['visual'].to(device)
                answer = sample['answer'].to(device)
                question_indexes = sample['question_id'] # keep in cpu
                if 'maskA' in sample:
                    mask_a = sample['maskA'].to(device)
                flag = sample['flag'].to(device)

                # get output
                if 'maskA' in sample:
                    output = model(options, visual, question, mask_a)
                else:
                    output = model(options, question)
                p.step()

if __name__ == '__main__':
    main()
    print("finished")
