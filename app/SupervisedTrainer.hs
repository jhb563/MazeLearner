import System.IO

import SupervisedLearner (runTraining)

main = do
  (w1, b1, w2, b2) <- runTraining "features_114.csv"
  handle <- openFile "supervised_weights.txt" WriteMode
  hPrint handle w1
  hPrint handle b1
  hPrint handle w2
  hPrint handle b2
  hClose handle
