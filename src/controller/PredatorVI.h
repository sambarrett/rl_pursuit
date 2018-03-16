#ifndef PREDATORVI_H_APZPFBYK
#define PREDATORVI_H_APZPFBYK

#include <armadillo>
#include "Agent.h"

class PredatorVI: public Agent {
public:
  PredatorVI(boost::shared_ptr<RNG> rng, const Point2D &dims, const std::string &filename);
  ActionProbs step(const Observation &obs);
  void restart();
  std::string generateDescription();
  Agent* clone();

protected:
  unsigned int calcStateInd(const Observation &obs);

  arma::fmat data;
  std::string filename;
};

#endif /* end of include guard: PREDATORVI_H_APZPFBYK */

