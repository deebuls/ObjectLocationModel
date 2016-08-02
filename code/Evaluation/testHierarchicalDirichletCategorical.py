import pandas as pd
from HierarchicalDirichletCategorical import HierarchicalDirichletCategorical


testdata = pd.DataFrame([[0,0],[1,1],[2,2]], columns=['time', 'location'])
modelH = HierarchicalDirichletCategorical(testdata)
modelH.create_model(model_type='all')


modelH.create_model(model_type='2fold')
