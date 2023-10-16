
from ViLD import ViLD
from ViLD_w_viz import ViLD_Viz

game_board_str = "square game board"
category_name_string = ';'.join(['black checkers piece', 'white checkers piece', game_board_str])
category_names = ['background'] + [x.strip() for x in category_name_string.split(';')]

v = ViLD() 
v.get_centroids('test1.jpg', category_names, game_board_str)

# v_test = ViLD_Viz()
# v_test.get_results('test1.jpg', category_names, game_board_str)