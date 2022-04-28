cd
source mc_env/bin/activate
kinit -l 240h && aklog  # will need to typ in pw after this
cd ~/multiagent-competition

ENV=you-shall-not-pass
#ENV=kick-and-defend
#ENV=sumo-humans

python train_mc_agents.py --env=${ENV} --ta_i=0 --n_train=200000000 --m_train=200
