
from .selection import prob_rank,equal,roulette_wheel,tournament
from .management import pop_greedy,ls_greedy,ls_sa

class Methods():
    def __init__(self,paras,problem) -> None:
        self.paras = paras      
        self.problem = problem
        if paras.selection == "prob_rank":
            self.select = prob_rank
        elif paras.selection == "equal":
            self.select = equal
        elif paras.selection == 'roulette_wheel':
            self.select = roulette_wheel
        elif paras.selection == 'tournament':
            self.select = tournament
        else:
            print("selection method "+paras.selection+" has not been implemented !")
            exit()

        if paras.management == "pop_greedy":
            self.manage = pop_greedy
        elif paras.management == 'ls_greedy':
            self.manage = ls_greedy
        elif paras.management == 'ls_sa':
            self.manage = ls_sa
        else:
            print("management method "+paras.management+" has not been implemented !")
            exit()

        
    def get_method(self):

        if self.paras.method == "ael":
            from .ael.ael import AEL
            return AEL(self.paras,self.problem,self.select,self.manage)
        elif self.paras.method == "eoh":   
            from .eoh.eoh import EOH
            return EOH(self.paras,self.problem,self.select,self.manage)
        elif self.paras.method in ['ls','sa']:   
            from .localsearch.ls import LS
            return LS(self.paras,self.problem,self.select,self.manage)
        elif self.paras.method == "funsearch":   
            from .funsearch.funsearch import FunSearch
            return FunSearch(self.paras,self.problem,self.select,self.manage)
        elif self.paras.method == "reevo":
            from .reevo.reevo import ReEVO
            return ReEVO(self.paras,self.problem,self.select,self.manage)
        else:
            print("method "+self.method+" has not been implemented!")
            exit()