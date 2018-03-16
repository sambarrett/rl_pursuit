#include <json/json.h>
#include <factory/WorldFactory.h>
#include <common/Util.h>
#include <iostream>
#include <cmath>
#include <boost/python.hpp>

void replaceOptsTrial(Json::Value &options, unsigned int trialNum, const std::string &student) {
  std::map<std::string,std::string> reps;
  reps["$(TRIALNUM)"] = boost::lexical_cast<std::string>(trialNum);
  Json::Value const &val = options["predatorOptions"];
  if (!val.isNull() && !val["student"].isNull()) {
    reps["$(STUDENT)"] = student;
  }
  jsonReplaceStrings(options,reps);
}

float KLDivergence(const std::vector<float> &P, const std::vector<float> &Q) {
  float total = 0;
  for (unsigned int i = 0; i < P.size(); i++) {
    if (P[i] < 1e-40)
      continue;
    if (Q[i] < 1e-42) {
      std::cerr << "KL divergence not defined " << P[i] << ", " << Q[i] << std::endl;
      exit(2);
    }
    total += P[i] * log(P[i] / Q[i]);
  }
  return total;
}

float JSDivergence(const std::vector<float> &P, const std::vector<float> &Q) {
  std::vector<float> M(P.size());
  for (unsigned int i = 0; i < P.size(); i++)
    M[i] = 0.5 * (P[i] + Q[i]);
  return 0.5 * (KLDivergence(P,M) + KLDivergence(Q,M));
}

float calcK(const std::vector<float> &T, const std::vector<float> &P) {
  //std::cout << "calcK: " << std::endl;
  //std::cout << "  " << T << std::endl;
  //std::cout << "  " << P << std::endl;
  std::vector<float> uniform(P.size(),1.0 / P.size());
  std::vector<float> point(P.size(),0);
  point[0] = 1.0;

  float val;

  float tp = JSDivergence(P,T);
  //std::cout << "  tp: " << tp << std::endl;
  if (tp < 1e-90) {
    val = 1.0;
  } else {
    float tuniform = JSDivergence(T,uniform);
    //std::cout << "  tuniform: " << tuniform << std::endl;
    if (tp < tuniform) {
      val = 1 - tp / tuniform;
    } else {
      //std::cout << "  puniform: " << JSDivergence(P,uniform) << std::endl;
      //std::cout << "  uniformPoint: " << JSDivergence(uniform,point) << std::endl;
      val = - JSDivergence(P,uniform) / JSDivergence(uniform,point);
    }
  }
  //std::cout << "  " << val << std::endl;
  return val;
}

bool incrementInds(std::vector<unsigned int> &inds, unsigned int maxInd) {
  unsigned int i = 0;
  while (true) {
    inds[i]++;
    if (inds[i] >= maxInd) {
      inds[i] = 0;
      i++;
      if (i == inds.size())
        return false;
      else
        continue;
    }
    
    // if we've reached here, the incrementing has succeeded
    return true;
  }
}

std::vector<ActionProbs> getTeammateActions(boost::shared_ptr<World> &world, boost::shared_ptr<AgentDummy> &adhocAgent, bool takeInitialStep, Action::Type initialAdhocAction) {
  std::vector<ActionProbs> actionProbList(5);
  world->loadPositions();
  world->restartAgents();
  adhocAgent->setAction(initialAdhocAction);
  if (takeInitialStep)
    world->step();

  world->step(actionProbList);
  
  boost::shared_ptr<WorldModel> model = world->getModel();
  int adhocInd = model->getAdhocInd();
  int preyInd = model->getPreyInd();
  std::vector<ActionProbs> teammateActionProbs;
  for (int i = 0; i < (int)actionProbList.size(); i++) {
    if ((i == adhocInd) || (i == preyInd))
      continue;
    teammateActionProbs.push_back(actionProbList[i]);
  }
  if (teammateActionProbs.size() != 3) {
    std::cerr << "Got too many teammates: " << teammateActionProbs.size() << " expected 3" << std::endl;
    exit(3);
  }

  return teammateActionProbs;
}

std::vector<std::vector<float> > getTeammateJointActions(boost::shared_ptr<World> &world, boost::shared_ptr<AgentDummy> &adhocAgent) {
  unsigned int numSharedActions = Action::NUM_ACTIONS * Action::NUM_ACTIONS * Action::NUM_ACTIONS;
  std::vector<std::vector<float> > actions(Action::NUM_ACTIONS,std::vector<float>(numSharedActions,0.0));

  for (unsigned int a = 0; a < Action::NUM_ACTIONS; a++) {
    std::vector<ActionProbs> teammateActionProbs = getTeammateActions(world,adhocAgent,true,(Action::Type)a);
    
    std::vector<unsigned int> agentInds(teammateActionProbs.size(),0);
    do {
      unsigned int jointAction = 0;
      float prob = 1.0;
      for (unsigned int i = 0; i < agentInds.size(); i++) {
        jointAction = Action::NUM_ACTIONS * jointAction + agentInds[i];
        prob *= teammateActionProbs[i][(Action::Type)agentInds[i]];
      }
      actions[a][jointAction] += prob;

    } while (incrementInds(agentInds,Action::NUM_ACTIONS));
  }
  return actions;
}

float calcReactivity(boost::shared_ptr<World> &world, boost::shared_ptr<AgentDummy> &adhocAgent) {
  std::vector<std::vector<float> > jointActionProbs = getTeammateJointActions(world,adhocAgent);

  float total = 0.0;

  for (unsigned int i = 0; i < Action::NUM_ACTIONS; i++) {
    for (unsigned int j = 0; j < Action::NUM_ACTIONS; j++) {
      if (i == j)
        continue;
      float val = JSDivergence(jointActionProbs[i],jointActionProbs[j]);
      total += val;
    }
  }

  unsigned int m = Action::NUM_ACTIONS;

  return 1.0 / (m * (m-1) * log(2)) * total;
}

float calcTeamK(boost::shared_ptr<World> &world, boost::shared_ptr<AgentDummy> &adhocAgent, std::vector<AgentPtr> &trueTeammates, std::vector<std::vector<AgentPtr> > &predTeammates) {
  // calculate the true actions
  std::vector<ActionProbs> trueTeammateActions;
  trueTeammateActions = getTeammateActions(world,adhocAgent,false,Action::NOOP); // NOOP doesn't matter

  // calculate the avg predicted actions
  std::vector<ActionProbs> predTeammateActions(trueTeammateActions.size());
  for (unsigned int modelInd = 0; modelInd < predTeammates.size(); modelInd++) {
    world->setAgentControllers(predTeammates[modelInd]);
    std::vector<ActionProbs> predTeammateActionsForModel = getTeammateActions(world,adhocAgent,false,Action::NOOP);
    for (unsigned int teammateInd = 0; teammateInd < predTeammateActionsForModel.size(); teammateInd++) {
      for (unsigned int a = 0; a < Action::NUM_ACTIONS; a++) {
        predTeammateActions[teammateInd][(Action::Type)a] += predTeammateActionsForModel[teammateInd][(Action::Type)a] / predTeammates.size();
      }
    }
  }
  
  // set the world back to a sane state
  world->setAgentControllers(trueTeammates);


  // calc the  val
  float total = 0;
  for (unsigned int i = 0; i < trueTeammateActions.size(); i++) { // loop over teammates
    // convert to std::vector<float>
    std::vector<float> trueActions(Action::NUM_ACTIONS);
    std::vector<float> predActions(Action::NUM_ACTIONS);
    for (unsigned int a = 0; a < Action::NUM_ACTIONS; a++) {
      trueActions[a] = trueTeammateActions[i][(Action::Type)a];
      predActions[a] = predTeammateActions[i][(Action::Type)a];
    }
    // do it
    total += calcK(trueActions,predActions);
  }

  return total / trueTeammateActions.size();
}

std::vector<std::string> getStudents(bool old, const std::string &excludedStudent = "") {
  std::string path = old ? "data/aamas11students.txt" : "data/newStudents29.txt";
  std::set<std::string> students;
  getAvailableStudents(path,students);
  students.erase(excludedStudent);
  std::vector<std::string> studentVec(students.begin(),students.end());
  return studentVec;
}

std::string selectRandomStudent(boost::shared_ptr<RNG> rng, bool old) {
  std::vector<std::string> students = getStudents(old);
  int ind = rng->randomInt(students.size());
  return students[ind];
}

void createWorldAndTeammates(boost::shared_ptr<World> &world, boost::shared_ptr<AgentDummy> &adhocAgent, std::vector<AgentPtr> &trueTeammates, std::vector<std::vector<AgentPtr> > &predTeammates, const Json::Value &options, unsigned int trialNum) {
  unsigned int randomSeed = getTime() * 1000000 + 1000 * getpid() + trialNum; // hopefully random enough
  bool output = options["verbosity"].get("description",false).asBool();
  Json::Value trialOptions(options);
  
  boost::shared_ptr<RNG> rng(new RNG (randomSeed));

  std::string student = selectRandomStudent(rng,trialOptions["predatorOptions"].get("old","").asBool());
  replaceOptsTrial(trialOptions,trialNum,student);

  Point2D dims = getDims(trialOptions);
  adhocAgent = boost::shared_ptr<AgentDummy>(new AgentDummy(rng,dims));

  std::vector<AgentModel> agentModels;
  int replacementInd = rng->randomInt(4);


  createAgentControllersAndModels(rng,dims,trialNum,replacementInd,trialOptions,adhocAgent,trueTeammates,agentModels);
  
  if (output)
    std::cout << "Models:" << std::endl;
  //predTeammates.resize(trialOptions["models"].size());
  for (unsigned int i = 0; i < trialOptions["models"].size(); i++) {
    if (trialOptions["models"][i]["predator"] == "student") {
      Json::Value predOpts = trialOptions["models"][i]["predatorOptions"];
      bool old = predOpts.get("old","").asBool();
      bool excluded = predOpts.get("exclude","").asBool();
      std::vector<std::string> students = getStudents(old,excluded ? student : "");
      for (unsigned int j = 0; j < students.size(); j++) {
        std::vector<AgentModel> tempAgentModels;
        if (output)
          std::cout << "  Model " << i << std::endl;
        trialOptions["predator"] = "dt";
        trialOptions["predatorOptions"] = predOpts;
        if (old) {
          std::cerr << "UNHANDLED OLD FOR NOW" << std::endl;
          exit(4);
        }
        trialOptions["predatorOptions"]["filename"] = std::string("data-school/dt/studentsNew29-unperturbed-50000/weighted/only-") + students[j] + ".weka";
        predTeammates.push_back(std::vector<AgentPtr>());
        createAgentControllersAndModels(rng,dims,trialNum,replacementInd,trialOptions,adhocAgent,predTeammates.back(),tempAgentModels);
        if (output) {
          for (unsigned int j = 0; j < predTeammates.back().size(); j++)
            std::cout << "    " << j << ": " << predTeammates.back()[j]->generateDescription() << std::endl;
          std::cout << "    -----" << std::endl;
        }
      }
    } else {
      std::vector<AgentModel> tempAgentModels;
      if (output)
        std::cout << "  Model " << i << std::endl;
      trialOptions["predator"] = trialOptions["models"][i]["predator"];
      trialOptions["predatorOptions"] = trialOptions["models"][i]["predatorOptions"];
      predTeammates.push_back(std::vector<AgentPtr>());
      createAgentControllersAndModels(rng,dims,trialNum,replacementInd,trialOptions,adhocAgent,predTeammates.back(),tempAgentModels);
      if (output) {
        for (unsigned int j = 0; j < predTeammates.back().size(); j++)
          std::cout << "    " << j << ": " << predTeammates.back()[j]->generateDescription() << std::endl;
        std::cout << "    -----" << std::endl;
      }
    }
  }
  if (output)
    std::cout << "  -----" << std::endl;

  boost::shared_ptr<WorldModel> worldModel = createWorldModel(dims);
  world = createWorld(rng,worldModel,0.0,true);
  if (output)
    std::cout << "TRUE:" << std::endl;
  for (unsigned int i = 0; i < trueTeammates.size(); i++) {
    if (output)
      std::cout << "  " << i << ": " << trueTeammates[i]->generateDescription() << std::endl;
    world->addAgent(agentModels[i],trueTeammates[i],true);
  }
  if (output)
    std::cout << "-----------" << std::endl;
}

int main(int argc, char *argv[])
{
  Json::Value options;
  std::string usage = "Usage: main config1 [config2 ...]";
  parseCommandLineArgs(&argc,&argv,usage,1,-1);

  for (int i = 1; i < argc; i++) {
    if (! readJson(argv[i],options)) {
      return 1;
    }
  }

  unsigned int numTrials = options.get("trials",1).asUInt();
  unsigned int numPointsPerTrial = options.get("pointsPerTrial",1).asUInt();
  unsigned int numTotalPoints = numTrials * numPointsPerTrial;

  float avgReactivity = 0;
  float avgTeamK = 0;

  unsigned int printFrequency = numTrials > 100? numTrials / 100.0 : 1;

  for (unsigned int trialNum = 0; trialNum < numTrials; trialNum++) {
    if (trialNum % printFrequency == 0)
      std::cout << "." << std::flush;
    boost::shared_ptr<World> world;
    boost::shared_ptr<AgentDummy> adhocAgent;
    std::vector<AgentPtr> trueTeammates;
    std::vector<std::vector<AgentPtr> > predTeammates;
    createWorldAndTeammates(world,adhocAgent,trueTeammates,predTeammates,options,trialNum);
    for (unsigned int i = 0; i < numPointsPerTrial; i++) {
      world->randomizePositions();
      world->storePositions();
      try {
        float reactivity = calcReactivity(world,adhocAgent) / numTotalPoints;
        float teamK = calcTeamK(world,adhocAgent,trueTeammates,predTeammates) / numTotalPoints;
        avgReactivity += reactivity;
        avgTeamK += teamK;
      } catch (boost::python::error_already_set) {
        continue;
      }
    }
  }
  std::cout << std::endl;

  std::cout << "For " << numTotalPoints << " sample points over " << numTrials << " trials" << std::endl;
  std::cout << "REACTIVITY: " << avgReactivity << std::endl;
  std::cout << "TEAMK     : " << avgTeamK << std::endl;

  return 0;
}
