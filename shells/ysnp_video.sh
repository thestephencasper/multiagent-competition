cd
source mc_env/bin/activate
cd ~/multiagent-competition

# https://github.com/openai/gym/issues/366
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python make_video.py --env=you-shall-not-pass --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=200m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=200m.sb
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python make_video.py --env=you-shall-not-pass --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=200m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=200m.sb
source ~/.bashrc
