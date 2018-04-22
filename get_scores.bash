#!/bin/bash

bestTrainScore="0"
bestTestScore="0"
bestTrainDir=""
bestTestDir=""
worstTrainScore="9.9999e+01"
worstTestScore="9.9999e+01"
worstTrainDir=""
worstTestDir=""
for i in logs/* ; do
  if [ -d "$i" ]; then
    echo "$i"
    trainScore=$(awk '{if(NR>6&&$1 > max) max = $1}END{print max}' RS=" " "$i/train.log")
    testScore=$(awk '{if(NR>6&&$1 > max) max = $1}END{print max}' RS=" " "$i/test.log")

    echo "Training: $trainScore"
    echo "Testing: $testScore"
    echo ""

    if [[ "$trainScore" > "$bestTrainScore" ]]; then
      bestTrainScore=$trainScore
      bestTrainDir=$i
    fi

    if [[ "$testScore" > "$bestTestScore" ]]; then
      bestTestScore=$testScore
      bestTestDir=$i
    fi

    if [[ "$trainScore" < "$worstTrainScore" ]]; then
      worstTrainScore=$trainScore
      worstTrainDir=$i
    fi

    if [[ "$testScore" < "$worstTestScore" ]]; then
      worstTestScore=$testScore
      worstTestDir=$i
    fi
  fi
done

echo "---------------------------------------------------"
#echo "Best Training Set: $bestTrainDir => $bestTrainScore"
echo "Best Testing Set: $bestTestDir => $bestTestScore"
echo "---------------------------------------------------"
#echo "Worst Training Set: $worstTrainDir => $worstTrainScore"
echo "Worst Testing Set: $worstTestDir => $worstTestScore"