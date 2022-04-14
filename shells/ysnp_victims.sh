cd
source mc_env/bin/activate
kinit -l 240h && aklog  # will need to typ in pw after this
cd ~/multiagent-competition

#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1

#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 --m_steps=40 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=20m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=20m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 --m_steps=40 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=20m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=20m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 --m_steps=40 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=20m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=20m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1 --m_steps=40 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=20m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=20m.sb

#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 --m_steps=60 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=40m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=40m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 --m_steps=60 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=40m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=40m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 --m_steps=60 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=40m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=40m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1 --m_steps=60 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=40m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=40m.sb

#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 --m_steps=80 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=60m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=60m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 --m_steps=80 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=60m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=60m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 --m_steps=80 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=60m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=60m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1 --m_steps=80 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=60m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=60m.sb

#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 --m_steps=100 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=80m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=80m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 --m_steps=100 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=80m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=80m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 --m_steps=100 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=80m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=80m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1 --m_steps=100 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=80m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=80m.sb

#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 --m_steps=120 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=100m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=100m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 --m_steps=120 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=100m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=100m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 --m_steps=120 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=100m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=100m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1 --m_steps=120 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=100m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=100m.sb

#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 --m_steps=140 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=120m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=120m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 --m_steps=140 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=120m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=120m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 --m_steps=140 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=120m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=120m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1 --m_steps=140 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=120m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=120m.sb

#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 --m_steps=160 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=140m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=140m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 --m_steps=160 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=140m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=140m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 --m_steps=160 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=140m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=140m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1 --m_steps=160 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=140m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=140m.sb

#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 --m_steps=180 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=160m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=160m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 --m_steps=180 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=160m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=160m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 --m_steps=180 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=160m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=160m.sb &
#python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1 --m_steps=180 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=160m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=160m.sb

python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=0 --m_steps=200 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=180m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=180m.sb &
python train_mc_agents.py --env=you-shall-not-pass --ta_i=0 --id=1 --m_steps=200 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=180m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=180m.sb &
python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=0 --m_steps=200 --agent0_ckpt=you-shall-not-pass_side=0_id=0_t=180m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=0_t=180m.sb &
python train_mc_agents.py --env=you-shall-not-pass --ta_i=1 --id=1 --m_steps=200 --agent0_ckpt=you-shall-not-pass_side=0_id=1_t=180m.sb --agent1_ckpt=you-shall-not-pass_side=1_id=1_t=180m.sb
