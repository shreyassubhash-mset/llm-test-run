import os, glob, base64
import pathlib, time
import numpy as np
import torch
import cv2
import pickle
import uvicorn
import jwt
from dotenv import load_dotenv

load_dotenv()

from shapely.geometry import Polygon
from random import shuffle
from datetime import datetime
from pytz import timezone
from fastapi.security import OAuth2PasswordBearer
from fastapi import HTTPException, Security
from fastapi import FastAPI, Form, Request, File, UploadFile, Form
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
# from detectron2.modeling import build_model
from ast import literal_eval
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

classes = ['N', '3', '4', 'K', 'Y', 'H', 'C', 'F', 'R', '7', 'J', 'X', 'W', '9', 'T', 'I', 'P', 'B', 'A']
labeledImagesPath = '/home/azureuser/detectron2/labeledImages/'
AUDIENCE = "https://us-uat3-ent.pampersrewards.com"

app = FastAPI()
# optional, required if you are serving static files
# app.mount("/static", StaticFiles(directory="static"), name="static")
# optional, required if you are serving webpage via template engine
templates = Jinja2Templates(directory="templates")
print("Successfully imported object detection libraries.")


pinMisspelled = ["PIN", "PPIN", "PPN", "PPTN", "PPI", "PI", "PP", "IN", "9IN", "9N", "N", "7IN", "NNN", "I", "PTN",
                 "TN", "PN", "PINN", "PIIN", "INN", "IIN", "PNN", "PITN", "PTIN", "4IN", "RN", "3PIN", "4PIN",
                 "FTN", "PTN", "FIN", "PTN", "P", "N", "I", "F", "T", "TN", "9TN", "P3N", "PN"]
os.makedirs(labeledImagesPath, exist_ok=True)


def decodeToken(token, secret=secret):
    try:
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail='Signature has expired')
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail='Invalid token')


def decodeTokenMultiple(token, secret=secret, secretAnother=secretNewUK):
    secretList = [secret, secretAnother]
    for secretSingle in secretList:
        if secretSingle != 'None':
            print("validating token using secret : ", secretSingle)
            try:
                payload = jwt.decode(token, secretSingle, algorithms=['HS256'])
                return payload
            except jwt.InvalidTokenError as e:
                print(e)
                pass
            except jwt.ExpiredSignatureError as e:
                print(e)
                pass
            except:
                print("error")

    print("none token passed validation")
    raise HTTPException(status_code=401, detail='Invalid token')
    

def decodeTokensWithAudience(token, audience_market, secret=secret, secretAnother=secretNewUK):
    secretList = [secret, secretAnother]
    for secretSingle in secretList:
        if secretSingle != 'None':
            print("validating token using secret : ", secretSingle)
            try:
                payload = jwt.decode(token, secretSingle, algorithms=['HS256'], audience=audience_market)
                return payload
            except jwt.InvalidTokenError as e:
                print(e)
                pass
            except jwt.ExpiredSignatureError as e:
                print(e)
                pass
            except:
                print("error")

    print("none token passed validation")
    raise HTTPException(status_code=401, detail='Invalid token')


def checkIfVertical(img):
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    if imgWidth < imgHeight:
        return True
    else:
        return False

def checkIfVerticals(img):
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    if imgWidth < imgHeight:
        return 'vertical'
    else:
        return 'horizontal'

def intersection_bbox(rect1, rect2):
    polygon = Polygon(rect1)
    other_polygon = Polygon(rect2)
    intersection = polygon.intersection(other_polygon)
    return intersection.area


def remove_items(test_list, item):
    # using filter() + __ne__ to perform the task
    res = list(filter((item).__ne__, test_list))

    return res


def pointLiesRectangle(bottomLeft, topRight, pt):
    if pt[0] > bottomLeft[0] and pt[0] < topRight[0] and pt[1] < bottomLeft[1] and pt[1] > topRight[1]:
        return 1
    else:
        return 0


def removeNonPinAlginedChars(linesList):
    yPrevMin = linesList[0][2][0][1]
    yPrevMax = linesList[0][2][2][1]
    width = yPrevMax - yPrevMin
    copy = linesList.copy()

    for id, charList in enumerate(linesList):
        if charList in linesList:
            if charList[2][0][1] >= yPrevMin - int(3.5 * width) and charList[2][2][1] <= yPrevMax + int(3.5 * width):
                yPrevMin = charList[2][0][1]
                yPrevMax = charList[2][2][1]
                width = yPrevMax - yPrevMin
            else:
                copy.remove(charList)

    return copy


def extractTextFromBoxes(linesList):
    result = ""
    temp = ""
    copy = linesList.copy()
    for id, charList in enumerate(linesList):
        if charList in linesList:
            rectHeight = int(charList[2][2][1]) - int(charList[2][1][1])
            rectWidth = int(charList[2][2][0]) - int(charList[2][0][0])
            centreRect = [(int(charList[2][0][0] + charList[2][1][0]) / 2),
                          (int(charList[2][0][1] + charList[2][3][1]) / 2)]
            leftIn = False
            rightIn = False
            centreIn = False
            heightUp = False
            heightDown = False

            # point right and left
            pointToIdentifyInRectangle = [centreRect[0], centreRect[1] + rectHeight]
            new_list = [pointLiesRectangle(charListX[2][3], charListX[2][1], pointToIdentifyInRectangle)
                        for charListX in linesList]
            pointToIdentifyInRectangleRight = [centreRect[0] + (rectWidth / 4), centreRect[1] + rectHeight]
            newList2 = [pointLiesRectangle(charListX[2][3], charListX[2][1], pointToIdentifyInRectangleRight)
                        for charListX in linesList]
            pointToIdentifyInRectangleLeft = [centreRect[0] - (rectWidth / 4), centreRect[1] + rectHeight]
            newList3 = [pointLiesRectangle(charListX[2][3], charListX[2][1], pointToIdentifyInRectangleLeft)
                        for charListX in linesList]
            pointToIdentifyInRectangleUp = [centreRect[0] - (rectWidth / 4), centreRect[1] + (rectHeight / 2)]
            newList4 = [pointLiesRectangle(charListX[2][3], charListX[2][1], pointToIdentifyInRectangleUp)
                        for charListX in linesList]
            pointToIdentifyInRectangleDown = [centreRect[0] - (rectWidth / 4), centreRect[1] + (1.5 * rectHeight)]
            newList5 = [pointLiesRectangle(charListX[2][3], charListX[2][1], pointToIdentifyInRectangleDown)
                        for charListX in linesList]

            if 1 in new_list:
                toAppend = linesList[new_list.index(1)] if linesList[new_list.index(1)][1] < charList[1] else charList
                temp += toAppend[0]
                centreIn = True
                try:
                    copy.remove(toAppend)
                except:
                    pass
            if centreIn == False and 1 in newList4:
                new_list = None
                new_list = newList4
                toAppend = linesList[new_list.index(1)] if linesList[new_list.index(1)][1] < charList[1] else charList
                temp += toAppend[0]
                heightUp = True
                centreIn = False
                try:
                    copy.remove(toAppend)
                except:
                    pass
            if centreIn == False and heightUp == False and 1 in newList5:
                new_list = None
                new_list = newList5
                toAppend = linesList[new_list.index(1)] if linesList[new_list.index(1)][1] < charList[1] else charList
                temp += toAppend[0]
                heightDown = True
                heightUp = False
                centreIn = False
                try:
                    copy.remove(toAppend)
                except:
                    pass
            if centreIn == False and heightUp == False and heightDown == False and 1 in newList2:
                new_list = None
                new_list = newList2
                toAppend = linesList[new_list.index(1)] if linesList[new_list.index(1)][1] < charList[1] else charList
                temp += toAppend[0]
                rightInin = True
                try:
                    copy.remove(toAppend)
                except:
                    pass
            if centreIn == False and rightIn == False and heightUp == False and heightDown == False and 1 in newList3:
                new_list = None
                new_list = newList3
                toAppend = linesList[new_list.index(1)] if linesList[new_list.index(1)][1] < charList[1] else charList
                temp += toAppend[0]
                leftIn = True
                rightInin = False
                centreIn = False
                try:
                    copy.remove(toAppend)

                except:
                    pass

    # run removeNonPinAlginedChars to remove nonPin outlier detections
    try:
        copy = removeNonPinAlginedChars(copy)
    except:
        pass
    result = "".join([charListX[0] for charListX in copy])

    return result


def exactCoordMatchPresent(linesList, coords, accuracy):
    present = None
    for id, charList in enumerate(linesList):
        if charList[2] == coords and charList[1] != accuracy:
            present = id
    return present


def removeDuplicateBoxes(linesList):
    result = linesList.copy()
    for id, charList in enumerate(linesList):
        coordsMatchId = exactCoordMatchPresent(linesList, charList[2], charList[1])
        if coordsMatchId is not None:
            minAccuracy = linesList[id] if linesList[id][1] < linesList[coordsMatchId][1] else linesList[coordsMatchId]
            if minAccuracy in result:
                result.remove(minAccuracy)

    for idx, charList in enumerate(result):
        if result[idx][0] != '-1':
            #             print(charList)
            # rect1 = [()]
            for idy in range(idx + 1, len(result) - 1):
                if result[idy][0] != '-1':
                    #                     print(result[idy][0], result[idy][1])
                    intersectionArea = intersection_bbox(result[idx][2], result[idy][2])
                    #                     print("intersection area : ", intersection_bbox(result[idx][2], result[idy][2]))
                    if intersectionArea > 300:
                        if result[idx][1] >= result[idy][1]:
                            result[idy] = ['-1', 0.0, [[0, 0], [0, 0], [0, 0], [0, 0]]]
                        else:
                            result[idx] = ['-1', 0.0, [[0, 0], [0, 0], [0, 0], [0, 0]]]
                        intersectionArea = 0.0
    result = remove_items(result, ['-1', 0.0, [[0, 0], [0, 0], [0, 0], [0, 0]]])
    return result


def readPickleFile(filePath):
    file = open(filePath, "rb")
    charCoordDict = pickle.load(file)

    return charCoordDict


def create_datatset():
    i = 0
    dataset_dicts = []
    for f_path, lines in df.groupby('img_p'):
        record = {}
        record["file_name"] = f_path
        record["image_id"] = i

        height, width = cv2.imread(f_path).shape[:2]
        record["height"] = height
        record["width"] = width
        objs = []

        for _, row in lines.iterrows():
            xmin, ymin, xmax, ymax = row.x, row.y, row.xm, row.ym
            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(classes.index(row.label)),
                "iscrowd": 0
            }

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
        i += 1
    return dataset_dicts


def initialise_model_config(model_path='./model', weight_filename="model_final.pth", classes=classes,
                            dataset_name="detect_box5"):
    try:
        DatasetCatalog.register(dataset_name, create_datatset)
    except:
        pass
    MetadataCatalog.get(dataset_name).set(thing_classes=classes)
    detect_metadata = MetadataCatalog.get(dataset_name)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ('')  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg['TEST']['AUG']['FLIP'] = False
    cfg.OUTPUT_DIR = model_path

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weight_filename)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = (dataset_name,)

    #     predictor = build_model(cfg)
    predictor = DefaultPredictor(cfg)
    #
    return predictor, detect_metadata


# Restore checkpoint
print('loading model... ', )
all_predictor, detect_metadata = initialise_model_config(
    './model21Oct10kIter5kDataLr0.0025Batch16Augmented0.23',
    weight_filename='model_final.pth')
print("\n\n\nmodel and parameters has been loaded into memory")


def get_orr_img_pred(im, predictor):
    img_pred = []
    outputs = predictor(im)
    predictions = outputs["instances"].to("cpu")
    boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classesDetected = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
    text_classes = [classes[i] for i in classesDetected]

    for idx, text in enumerate(text_classes):
        boxAppend = boxes.astype(int).tolist()[idx]
        img_pred.append([text, scores[idx].item(), [[boxAppend[0], boxAppend[1]], [boxAppend[2], boxAppend[1]],
                                                    [boxAppend[2], boxAppend[3]], [boxAppend[0], boxAppend[3]]]])
    img_pred = sorted(img_pred, key=lambda x: x[2][0], reverse=False)

    return img_pred


def postProcessPredsToString(imgName, linesList):
    if linesList is not None:
        newLinesList = []
        if imgName:  # == "29076793-A981-4CE1-B530-0A1F8C43284D.jpg":
            newLinesList = removeDuplicateBoxes(linesList)
            # print("duplicates removed")
            tempStr = extractTextFromBoxes(newLinesList)
            idx = 0
            stringToReplace = ""
            for pin in pinMisspelled:
                # print("searching in : ", tempStr[:4])
                if pin in tempStr[:4]:
                    subPin = tempStr.find(pin)
                    if len(pin) > len(stringToReplace):
                        idx = subPin
                        stringToReplace = pin
            # print(idx, stringToReplace)

        if len(stringToReplace) != 0:
            # print("---------------------> before : ", tempStr)
            tempStr = tempStr[0:len(stringToReplace)].replace(stringToReplace, "PIN") + tempStr[len(stringToReplace):]
            tempStr = tempStr.replace("PIN", "")
            # print("--------------------> result : ", tempStr)

    print("imgName : {}".format(imgName))
    print("String : {}".format(tempStr))
    # print("Matching PIN : {}".format(stringToReplace))

    return tempStr


def returnResult(imageArrayPath, imageData, isVertical=False):
    img = np.fromstring(imageData, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if isVertical:
        checkVertical = checkIfVertical(img)
        if checkVertical:
            out = cv2.transpose(img)
            img = cv2.flip(out, flipCode=1)

    img_pred = get_orr_img_pred(img, all_predictor)
    result = postProcessPredsToString(imageArrayPath, img_pred)

    return result

def returnResults(imageArrayPath, imageData, isVertical):
    img = np.fromstring(imageData, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    checkVertical = checkIfVerticals(img)
    if checkVertical == 'vertical':
        out = cv2.transpose(img)
        img = cv2.flip(out, flipCode=1)

    img_pred = get_orr_img_pred(img, all_predictor)
    result = postProcessPredsToString(imageArrayPath, img_pred)

    return result


def main():
    imgList = glob.glob(imagesTestPath)
    print(imgList[5])
    start = time.clock()
    result = returnResult(imgList[5])
    end = time.clock()
    print("Time per image: {} ".format((end - start)))


# depends on use cases
class Item(BaseModel):
    language: str = 'english'

def getToken(program_code):
    # Load token from environment files
    shellScriptFile = 'secretKeys.sh'

    with open(shellScriptFile, 'r') as tokenFile:
        # Filter lines starting with 'export'
        envLines = [line.strip() for line in tokenFile.readlines() if line.strip().startswith('export')]

    # Create a dictionary from environment variable assignments
    tokenEnv = {}
    for line in envLines:
        key, value = map(str.strip, line.split('='))
        value = value.strip('"')  # Strip double quotes from the value
        tokenEnv[key] = value


    # Get the audience
    tokenKey = f'export {program_code}_JWT_SECRET'

    # Check if the audience key is in the environment
    if tokenKey not in tokenEnv:
        raise KeyError(f"Key '{tokenKey}' not found in the shell script file.")

    # Return the audience value
    return tokenEnv[tokenKey]

def getAudience(program_code):
    # Load audience from shell script file
    shellScriptFile = 'secretKeys.sh'

    with open(shellScriptFile, 'r') as audienceFile:
        # Filter lines starting with 'export'
        envLines = [line.strip() for line in audienceFile.readlines() if line.strip().startswith('export')]

    # Create a dictionary from environment variable assignments
    audienceEnv = {}
    for line in envLines:
        key, value = map(str.strip, line.split('='))
        value = value.strip('"')  # Strip double quotes from the value
        audienceEnv[key] = value

    # Get the audience
    audienceKey = f'export {program_code}_AUDIENCE'

    # Check if the audience key is in the environment
    if audienceKey not in audienceEnv:
        raise KeyError(f"Key '{audienceKey}' not found in the shell script file.")

    # Return the audience value
    return audienceEnv[audienceKey]


@app.get("/home/", response_class=HTMLResponse)
def writeHome(request: Request):
    return templates.TemplateResponse("fileUploadFastApi.html", {"request": request})


@app.post('/uploadImageUK')
# def handleForm(imageFile: UploadFile = File(...), token: str = Security(tokenCheckUK)):
def handleForm(imageFile: UploadFile = File(...)):
    # if decodeTokenMultiple(token, secret=secret, secretAnother=secretNewUK):
    print("----------------------------\nUK")
    img = imageFile.file.read()
    imgName = imageFile.filename
    result = returnResult(imgName, img)

    return {"result": result}

    # else:
    #     return {"result": "token not authorized"}


@app.post('/uploadImageFR')
# def handleForm(imageFile: UploadFile = File(...), token: str = Security(tokenCheckFR)):
def handleForm(imageFile: UploadFile = File(...)):
    # if decodeTokenMultiple(token, secret=secret, secretAnother=secretNewFR):
    print("----------------------------\nFR")
    img = imageFile.file.read()
    imgName = imageFile.filename
    result = returnResult(imgName, img)

    return {"result": result}

    # else:
    #     return {"result": "token not authorized"}


@app.post('/uploadImageDE')
# def handleForm(imageFile: UploadFile = File(...), token: str = Security(tokenCheckDE)):
def handleForm(imageFile: UploadFile = File(...)):
    # if decodeTokenMultiple(token, secret=secret, secretAnother=secretNewDE):
    print("----------------------------\nDE")
    img = imageFile.file.read()
    imgName = imageFile.filename
    result = returnResult(imgName, img)

    return {"result": result}




@app.post('/EURBU')
def handleForm(imageFile: UploadFile = File(...), token: str = Security(tokenCheckAT)):
    print("----------------------------\nAT Token: ", token)
    if decodeTokensWithAudience(token, audienceAT, secret=secretAT, secretAnother='None'):
        img = imageFile.file.read()
        imgName = imageFile.filename
        result = returnResult(imgName, img)

        return {"result": result}
    else:
        return {"result": "token not authorized"}


@app.post('/EURBUVertical')
def handleForm(imageFile: UploadFile = File(...), token: str = Security(tokenCheckAT)):
    print("----------------------------\nAT Vertical Token: ", token)
    if decodeTokensWithAudience(token, audienceAT, secret=secretAT, secretAnother='None'):
        img = imageFile.file.read()
        imgName = imageFile.filename
        result = returnResult(imgName, img, True)

        return {"result": result}
    else:
        return {"result": "token not authorized"}
    
@app.post('/uploadImage')
def handleForm(imageFile: UploadFile = File(...), token: str = Security(tokenCheck), orientation: str = "horizontal", programcode: str = None):

    secret = getToken(programcode)
    audience = getAudience(programcode)

    print(f"----------------------------\n{programcode} Token: ", token)
    if decodeTokensWithAudience(token, audience, secret, secretAnother='None'):
        img = imageFile.file.read()
        imgName = imageFile.filename
        result = returnResults(imgName, img, orientation)

        return {"result": result}
    else:
        return {"result": "token not authorized"}

# @app.post('/uploadImageNoToken')
# async def getResults(imageFile: UploadFile = File(...)):
#     format = "%Y-%m-%d %H:%M:%S %Z%z"
#     now_utc = datetime.now(timezone('UTC'))
#     now_asia = now_utc.astimezone(timezone('Asia/Kolkata'))

#     img = imageFile.file.read()
# print
#     imgName = imageFile.filename
#     img1 = np.fromstring(img, np.uint8)
#     img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
#     fPath = os.path.join(UPLOAD_FOLDER, now_asia.strftime(format)+os.path.basename(imgName))
#     fPath = fPath.replace(" ", "")
#     cv2.imwrite(fPath, img1)
#     result = returnResult(fPath)

#     try:
#         os.remove(fPath)
#     except:
#         pass
#     img = None
#     img1 = None
#     jpg_original = None
#     fPath = ""

#     return {"result": result}


if __name__ == '__main__':
    uvicorn.run('fastApiServer:app', host='0.0.0.0', port=8000)
