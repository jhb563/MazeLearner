{-# LANGUAGE ScopedTypeVariables #-}

module BasicLearner where

import Control.Monad.IO.Class (liftIO)
import Control.Monad.State
import Data.Int (Int32)
import Data.Vector (Vector(..), toList)
import qualified Data.Vector as V
import System.Random (getStdGen, randomR, StdGen)
import System.IO
import TensorFlow.Core
import TensorFlow.Minimize
import TensorFlow.Ops hiding (initializedVariable)
import TensorFlow.Variable

import LearnerLib.Serialization
import Runner (stepWorld, updateEnvironment)
import Types

data Model = Model
  { weightsT :: Variable Float
  , iterateWorldStep :: TensorData Float -> Session (Vector Float)
  , trainStep :: TensorData Float -> TensorData Float -> Session ()
  }

buildModel :: Maybe (Vector Float) -> Session Model
buildModel inputWeights = do
  initialWeights <- case inputWeights of
    Nothing -> truncatedNormal (vector [15, 10])
    Just w -> render $ constant (Shape [15, 10]) (toList w)
  -- This is all we need for the first session run
  (inputs :: Tensor Value Float) <- placeholder (Shape [1, 15])
  (weights :: Variable Float) <- initializedVariable initialWeights
  let (allOutputs :: Tensor Build Float) = inputs `matMul` (readValue weights)
  returnedOutputs <- render allOutputs

  -- Now for the second graph
  (nextOutputs :: Tensor Value Float) <- placeholder (Shape [10, 1])
  let (diff :: Tensor Build Float) = nextOutputs `sub` allOutputs
  let (loss :: Tensor Build Float) = reduceSum (diff `mul` diff)
  trainer_ <- minimizeWith adam loss [weights]
  let iterateStep = \inputFeed -> runWithFeeds [feed inputs inputFeed] returnedOutputs
  let trainingStep = \inputFeed nextOutputFeed -> runWithFeeds
                        [ feed inputs inputFeed
                        , feed nextOutputs nextOutputFeed
                        ]
                        trainer_
  return $ Model
    weights
    iterateStep
    trainingStep

runAllIterations :: Model -> World -> StateT ([Float], Int, Float) Session ()
runAllIterations model initialWorld = do
  let numIterations = 2000
  handle <- liftIO $ openFile "results.txt" WriteMode
  forM [1..numIterations] $ \i -> do
    liftIO $ print i
    liftIO $ hPrint handle i
    gen <- liftIO getStdGen
    (wonGame, (_, finalReward, _)) <- runStateT (runWorldIteration handle model) (initialWorld, 0.0, gen)
    (prevRewards, prevWinCount, randomChance) <- get
    let newRewards = finalReward : prevRewards
    let newWinCount = if wonGame then prevWinCount + 1 else prevWinCount
    let modifiedRandomChance = 1.0 / ((fromIntegral i / 40.0) + 5)
    put (newRewards, newWinCount, modifiedRandomChance)
  liftIO $ hClose handle
  return ()

runWorldIteration
  :: Handle
  -> Model
  -> StateT (World, Float, StdGen) (StateT ([Float], Int, Float) Session) Bool
runWorldIteration handle model = do
  (prevWorld :: World, prevReward, gen) <- get
  (_, _, randomChance) <- lift get
  -- Vectorize World and get move, then run to next world state
  let inputWorldVector = encodeTensorData (Shape [1, 15]) (vectorizeWorld prevWorld)
  (currentMoveWeights :: Vector Float) <- lift $ lift $ (iterateWorldStep model) inputWorldVector

  -- At Random Interval, explore new action
  let bestMove = moveFromOutput currentMoveWeights
  let (newMove, newMoveIndex, newGen) = chooseMoveWithRandomChance (bestMove, V.maxIndex currentMoveWeights) gen randomChance

  -- Get new World based on the ouput
  let (nextWorld, moveWasValid) = stepWorld newMove prevWorld
  let validMoveExtra = if moveWasValid then 0.01 else -0.01
  when (worldResult nextWorld == GameLost) $ do
    let s = "Lost! " ++ show (worldTime nextWorld) ++ show (playerLocation . worldPlayer $ nextWorld)
    liftIO (putStrLn s)
    liftIO $ hPutStrLn handle s
  when (worldResult nextWorld == GameWon) $ do
    liftIO (putStrLn "Won!")
    liftIO $ hPutStrLn handle "Won!"
  when (worldTime nextWorld > 1000) $ do
    liftIO (putStrLn "Time out!")
    liftIO $ hPutStrLn handle "Time out!"
  let (newReward, continuationAction) = case worldResult nextWorld of
        GameInProgress -> if worldTime nextWorld > 1000 then (-1.0, return False) else (0.0 + validMoveExtra, runWorldIteration handle model)
        GameWon -> (1.0, return True)
        GameLost -> (-1.0, return False)
  -- Get next action values
  let nextWorldVector = encodeTensorData (Shape [1, 15]) (vectorizeWorld nextWorld)
  (nextMoveVector :: Vector Float) <- lift $ lift $ (iterateWorldStep model) nextWorldVector
  let maxScore = V.maximum nextMoveVector
  let (targetActionValues :: Vector Float) =
        nextMoveVector V.// [(newMoveIndex, newReward + (0.99 * maxScore))]
  let targetActionData = encodeTensorData (Shape [10, 1]) targetActionValues
  lift $ lift $ (trainStep model) inputWorldVector targetActionData
  put (nextWorld, prevReward + newReward, newGen)
  continuationAction
  where
    chooseMoveWithRandomChance :: (PlayerMove, Int) -> StdGen -> Float -> (PlayerMove, Int, StdGen)
    chooseMoveWithRandomChance (bestMove, bestMoveIndex) gen randomChance =
      let (randVal, gen') = randomR (0.0, 1.0) gen
          (randomIndex, gen'') = randomR (0, 1) gen'
          randomMove = moveFromIndex randomIndex
      in  if randVal < randomChance
            then (randomMove, randomIndex, gen'')
            else (bestMove, bestMoveIndex, gen')

trainGame :: World -> Session (Int, Vector Float)
trainGame w = do
  (weights :: Vector Float) <- read <$> liftIO (readFile "weights_2.txt")
  model <- buildModel Nothing
  let initialRandomChance = 0.2
  (finalReward, finalWinCount, _) <- execStateT (runAllIterations model w) ([], 0, initialRandomChance)
  weights <- run (readValue $ weightsT model)
  return (finalWinCount, weights)

playGameTraining :: World -> IO (Int, Vector Float)
playGameTraining w = runSession (trainGame w)
