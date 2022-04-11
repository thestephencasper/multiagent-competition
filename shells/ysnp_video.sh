cd
source mc_env/bin/activate
#kinit -l 240h && aklog  # will need to typ in pw after this
cd ~/multiagent-competition

unset LD_PRELOAD
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
python make_video.py --env=you-shall-not-pass --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=40m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=40m.sb
#python train_mc_agents.py --env=you-shall-not-pass --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=100m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=100m.sb
source ~/.bashrc
