from integration import Integrator

i = Integrator()
i.initialize_board_state('test1.jpg')
move = i.query_LLM()
print(move) 