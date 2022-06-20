# importing the eel library  
import eel
import test
import playVideo
import constants
# initializing the application
eel.init("myWeb")


# using the eel.expose command
@eel.expose
# defining the function for addition of two numbers  
def handleTest(path):
    return test.test(path)

@eel.expose
def playVid(path):
    playVideo.play(path)

@eel.expose
def defPath():
    return constants.Root_Path

@eel.expose
def handleTrain():
    int1 = int(data_1)
    int2 = int(data_2)
    output = int1 + int2
    return output

# starting the application
eel.init('UI')
eel.start("index.html")