import os
import time
import numpy as np
import playsound as ps

def Voice(pre):
	if pre == "Afar":
		print("[INFO] start playing Afar music")
		ps.playsound(r'./Voice_data/')
		time.sleep(5)
		print("Well-Done Afar")

	elif pre == "Ahmara":
		print("[INFO] start playing Ahmara music")
		ps.playsound('./Voice_data/')
		time.sleep(5)
		print("Well-Done Ahmara")

	elif pre == "Gurage":
		print("[INFO] start playing Gurage music")
		ps.playsound('./Voice_data/')
		time.sleep(5)
		print("Well-Done Gurage")
	
	elif pre == "Oromifa":
		print("[INFO] start playing Oromifa music")
		ps.playsound('./Voice_data/')
		time.sleep(5)
		print("Well-Done Oromifa")

	elif pre == "Tigre":
		print("[INFO] start playing Tigre music")
		ps.playsound('./Voice_data/')
		time.sleep(5)
		print("Well-Done Tigre")

	elif pre == "Wolaita":
		print("[INFO] start playing Wolaita music")
		ps.playsound('./Voice_data/')
		time.sleep(5)
		print("Well-Done Wolaita")
