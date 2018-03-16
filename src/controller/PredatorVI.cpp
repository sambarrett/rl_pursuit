#include "PredatorVI.h"

PredatorVI::PredatorVI(boost::shared_ptr<RNG> rng, const Point2D &dims, const std::string &filename):
  Agent(rng,dims),
  filename(filename)
{
  bool res = data.load(filename);
  if (!res) {
    std::cerr << "PredatorVI: Error loading " << filename << std::endl;
    exit(2);
  }
}

void PredatorVI::restart() {
}

Agent* PredatorVI::clone() {
  return new PredatorVI(*this);
}

std::string PredatorVI::generateDescription() {
  return "PredatorVI for " + filename;
}

ActionProbs PredatorVI::step(const Observation &obs) {
  ActionProbs action;
  unsigned int stateInd = calcStateInd(obs);
  unsigned int maxA = 6;
  arma::frowvec vec = data.row(stateInd);
  maxA = vec.max();
  switch (maxA) {
    case 0:
      action[Action::NOOP] = 1.0;
      break;
    case 1:
      action[Action::RIGHT] = 1.0;
      break;
    case 2:
      action[Action::LEFT] = 1.0;
      break;
    case 3:
      action[Action::UP] = 1.0;
      break;
    case 4:
      action[Action::DOWN] = 1.0;
      break;
    default:
      std::cerr << "got bad max action: " << maxA << std::endl;
      exit(99);
      break;
  }
  return action;
}
  
unsigned int PredatorVI::calcStateInd(const Observation &obs) {
  unsigned int stateInd = 0;
  for (int i = obs.positions.size()-1; i >= 0; i--) {
    if ((i == (int)obs.preyInd) || (i == (int)obs.myInd))
      continue;
    stateInd *= dims.y;
    stateInd += obs.positions[i].y;
    stateInd *= dims.x;
    stateInd += obs.positions[i].x;
  }
  stateInd *= dims.y;
  stateInd += obs.myPos().y;
  stateInd *= dims.x;
  stateInd += obs.myPos().x;

  return stateInd;
}
