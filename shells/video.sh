
source ~/mc_env/bin/activate
cd ~/multiagent-competition

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python make_video.py --env=kick-and-defend --agent0_ckpt=kick-and-defend_side=0_id=0_t=40m --agent1_ckpt=kick-and-defend_side=1_id=0_t=40m
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python make_video.py --env=kick-and-defend --agent0_ckpt=kick-and-defend_side=0_id=1_t=40m --agent1_ckpt=kick-and-defend_side=1_id=1_t=40m

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python make_video.py --env=kick-and-defend --agent0_ckpt=kick-and-defend_side=0_id=0_t=130m --agent1_ckpt=kick-and-defend_side=1_id=0_t=130m
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python make_video.py --env=kick-and-defend --agent0_ckpt=kick-and-defend_side=0_id=1_t=130m --agent1_ckpt=kick-and-defend_side=1_id=1_t=130m
