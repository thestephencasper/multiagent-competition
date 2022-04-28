cd
source mc_env/bin/activate
kinit -l 240h && aklog  # will need to typ in pw after this
cd ~/multiagent-competition

START=0
STEP=2  # make sure the same as args.n_train in utils.py
AT=$START
STOP=1000
SLEEP=100
#ENV=you-shall-not-pass
ENV=kick-and-defend

if [ $START == 0 ]
then
  python train_mc_agents.py --env=${ENV} --ta_i=1 --id=0 &
  python train_mc_agents.py --env=${ENV} --ta_i=0 --id=0 &
  python train_mc_agents.py --env=${ENV} --ta_i=1 --id=1 &
  python train_mc_agents.py --env=${ENV} --ta_i=0 --id=1
  AT=$[STEP]
fi

while [ $AT -lt $STOP ]
do
  NEXT=$[AT+STEP]
  python train_mc_agents.py --env=${ENV} --ta_i=1 --id=0 --m_train=${NEXT} --agent0_ckpt=${ENV}_side=0_id=sample_id_t=sample_opptm --agent1_ckpt=${ENV}_side=1_id=0_t=${AT}m &
  python train_mc_agents.py --env=${ENV} --ta_i=0 --id=0 --m_train=${NEXT} --agent0_ckpt=${ENV}_side=0_id=0_t=${AT}m --agent1_ckpt=${ENV}_side=1_id=sample_id_t=sample_opptm &
  python train_mc_agents.py --env=${ENV} --ta_i=1 --id=1 --m_train=${NEXT} --agent0_ckpt=${ENV}_side=0_id=sample_id_t=sample_opptm --agent1_ckpt=${ENV}_side=1_id=0_t=${AT}m &
  python train_mc_agents.py --env=${ENV} --ta_i=0 --id=1 --m_train=${NEXT} --agent0_ckpt=${ENV}_side=0_id=0_t=${AT}m --agent1_ckpt=${ENV}_side=1_id=sample_id_t=sample_opptm || break
  sleep $SLEEP
  AT=$NEXT
done
