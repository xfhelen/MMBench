def run_cmd( cmd_str='', echo_print=1):
   
    from subprocess import run
    if echo_print == 1:
        print('\n执行cmd指令="{}"'.format(cmd_str))
    run(cmd_str, shell=True)

str='mkdir model_ckpt ' \
    'wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip -P model_ckpt ' \
    'unzip model_ckpt/models_2022.zip -d model_ckpt/ ' \
    'rm model_ckpt/models_2022.zip'


print("downloading the Pre-trained agent files for all 4 methods")
run_cmd('/home/niumo/transfuser-2022/team_code_transfuser/evaluation/evaluation_script.sh')
print('dowmload completed')