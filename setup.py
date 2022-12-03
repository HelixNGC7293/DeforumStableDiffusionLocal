#@markdown **Setup Environment**

setup_environment = True #@param {type:"boolean"}
print_subprocess = False #@param {type:"boolean"}
use_xformers_for_colab = False

if setup_environment:
    import subprocess, time, sys
    import os
    print("Setting up environment...")
    start_time = time.time()
    os.system("conda install git -y")
    os.system("conda install -c conda-forge opencv -y")
    os.system("conda install -c conda-forge ffmpeg -y")
    print("Setting up environment part2...")
    all_process = [
        ['pip', 'install', 'torch==1.12.1+cu113', 'torchvision==0.13.1+cu113', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'],
        ['pip', 'install', 'omegaconf==2.2.3', 'einops==0.4.1', 'pytorch-lightning==1.7.4', 'torchmetrics==0.9.3', 'torchtext==0.13.1', 'transformers==4.21.2', 'kornia==0.6.7'],
        ['git', 'clone', 'https://github.com/deforum-art/deforum-stable-diffusion'],
        ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq','scikit-learn'],
        ['pip', 'install', 'IPython'],
        ['pip', 'install', 'pandas'],
        ['pip', 'install', 'scikit-image'],
        ['pip', 'uninstall', 'numpy', '-y'],
        ['pip', 'install', '-U', 'numpy'],
        ['pip', 'install', 'opencv-contrib-python'],
        ['pip', 'install', 'numexpr']
    ]
    for process in all_process:
        running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
        if print_subprocess:
            print(running)
    
    with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
            f.write('')
    sys.path.extend([
        'deforum-stable-diffusion/',
        'deforum-stable-diffusion/src',
    ])
    end_time = time.time()

    if use_xformers_for_colab:

        print("..installing xformers")

        all_process = [['pip', 'install', 'triton==2.0.0.dev20220701']]
        for process in all_process:
            running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
            if print_subprocess:
                print(running)
                
        v_card_name = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        if 't4' in v_card_name.lower():
            name_to_download = 'T4'
        elif 'v100' in v_card_name.lower():
            name_to_download = 'V100'
        elif 'a100' in v_card_name.lower():
            name_to_download = 'A100'
        elif 'p100' in v_card_name.lower():
            name_to_download = 'P100'
        else:
            print(v_card_name + ' is currently not supported with xformers flash attention in deforum!')

        x_ver = 'xformers-0.0.13.dev0-py3-none-any.whl'
        x_link = 'https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/' + name_to_download + '/' + x_ver
    
        all_process = [
            ['wget', x_link],
            ['pip', 'install', x_ver],
            ['mv', 'deforum-stable-diffusion/src/ldm/modules/attention.py', 'deforum-stable-diffusion/src/ldm/modules/attention_backup.py'],
            ['mv', 'deforum-stable-diffusion/src/ldm/modules/attention_xformers.py', 'deforum-stable-diffusion/src/ldm/modules/attention.py']
        ]

        for process in all_process:
            running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
            if print_subprocess:
                print(running)

    print(f"Environment set up in {end_time-start_time:.0f} seconds")