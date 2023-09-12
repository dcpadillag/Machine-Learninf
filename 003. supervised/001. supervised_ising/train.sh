rm -r data
rm -r temp
#rm -r gif

mkdir data
mkdir temp
#mkdir gif

N=$1

python lattice.py $N

for i in `seq 1 1 50`
do
echo $i
time python wolff_cluster.py $i $N
done
