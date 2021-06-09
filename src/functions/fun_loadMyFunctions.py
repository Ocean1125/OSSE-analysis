

#def loadMyFunctions():
 #   import src.functions.fun_loadStartup as loadStartup
from src.functions.fun_loadRomsGrid import loadRomsGrid
from src.functions.fun_loadRomsGridFromHist import loadRomsGridFromHist
from src.functions.fun_loadCartopyCoast import loadCartopyCoast
Coast = loadCartopyCoast()

print('loaded my own functions')