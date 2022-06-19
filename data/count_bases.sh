total_lines=0
for i in {0001..2000}
do
	id=$(printf %04d $i)
	fn="chiron/lambda/eval/all/Lambda_eval_${id}.label"
	lines=$(cat $fn | wc -l)
	total_lines=$(($total_lines + $lines))
done
echo $total_lines
