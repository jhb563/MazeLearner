{-# LANGUAGE ScopedTypeVariables #-}

module BasicLearner where

import Control.Monad.IO.Class (liftIO)
import Control.Monad.State
import Data.Int (Int32)
import Data.Vector (Vector(..))
import qualified Data.Vector as V
import TensorFlow.Core
import TensorFlow.Minimize
import TensorFlow.Ops hiding (initializedVariable)
import TensorFlow.Variable

import Serialization
import Runner (stepWorld, updateEnvironment)
import Types

data Model = Model
  { weightsT :: Variable Float
  , iterateWorldStep :: TensorData Float -> Session (Vector Float)
  , trainStep :: TensorData Float -> TensorData Float -> Session ()
  }

buildModel :: Session Model
buildModel = do
  (initialWeights :: Tensor Value Float) <- truncatedNormal (vector [8, 10])
  -- This is all we need for the first session run
  (inputs :: Tensor Value Float) <- placeholder (Shape [1, 8])
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

runAllIterations :: Model -> World -> StateT ([Float], Int) Session ()
runAllIterations model initialWorld = do
  let numIterations = 2000
  forM [1..numIterations] $ \i -> do
    (wonGame, (_, finalReward)) <- runStateT (runWorldIteration model) (initialWorld, 0.0)
    (prevRewards, prevWinCount) <- get
    let newRewards = finalReward : prevRewards
    let newWinCount = if wonGame then prevWinCount + 1 else prevWinCount
    put (newRewards, newWinCount)
    -- TODO Reduce chance of random action over time
  return ()

runWorldIteration
  :: Model
  -> StateT (World, Float) (StateT ([Float], Int) Session) Bool
runWorldIteration model = do
  (prevWorld :: World, prevReward) <- get
  -- Vectorize World and get move, then run to next world state
  let inputWorldVector = encodeTensorData (Shape [1, 8]) (vectorizeWorld prevWorld)
  (currentMove :: Vector Float) <- lift $ lift $ (iterateWorldStep model) inputWorldVector
  -- Get new World based on the ouput
  let newMove = moveFromOutput currentMove
  -- TODO At Random Interval, explore new action
  let nextWorld = updateEnvironment (stepWorld newMove prevWorld)
  let (newReward, continuationAction) = case worldResult nextWorld of
        GameInProgress -> (0.0, runWorldIteration model)
        GameWon -> (1.0, return True)
        GameLost -> (-1.0, return False)
  -- Get next action values
  let nextWorldVector = encodeTensorData (Shape [1, 8]) (vectorizeWorld nextWorld)
  (nextMoveVector :: Vector Float) <- lift $ lift $ (iterateWorldStep model) nextWorldVector
  let (bestNextMoveIndex, maxScore) = (V.maxIndex nextMoveVector, V.maximum nextMoveVector)
  let (targetActionValues :: Vector Float) =
        nextMoveVector V.// [(bestNextMoveIndex, newReward + (0.99 * maxScore))]
  let targetActionData = encodeTensorData (Shape [10, 1]) targetActionValues
  lift $ lift $ (trainStep model) nextWorldVector targetActionData
  put (nextWorld, prevReward + newReward)
  continuationAction

trainGame :: World -> Session (Vector Float)
trainGame w = do
  model <- buildModel
  (finalReward, finalWinCount) <- execStateT (runAllIterations model w) ([], 0)
  run (readValue $ weightsT model)

playGameTraining :: World -> IO (Vector Float)
playGameTraining w = runSession (trainGame w)
