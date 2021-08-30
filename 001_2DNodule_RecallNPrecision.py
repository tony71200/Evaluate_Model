import json
import csv
from decimal import Decimal, ROUND_HALF_UP
import os

#ModelName = '001_Original'
#ModelName = '002_2xinteration'
#ModelName = '003_LateralCsp'
#ModelName = '004_Lateralc1c2'
#ModelName = '004_Lateralc1c2full'
# ModelName = '005_Lateralc1c2Sppx2'
#ModelName = '006_yolov4_Origin'
# ModelName = '007_yolov4_4P'
# ModelName = '009_EfficientD4'
ModelName = '010_EfficientDet_D1'
GroundTruthName = '000_GroundTruth'

ConfScore = ['0.01', '0.01']
cs_max = 1 # No confidence score = 0.01
TestName = ['A', 'B', 'C', 'Me']
iou_thres = 0.5
iop_thres = 0.7
iogt_thres = 0.7
TestTpW = [0, 0, 0, 0]
TestTpWo = [0, 0, 0, 0]
TestFpW = [0, 0, 0, 0]
TestFpWo = [0, 0, 0, 0]
TestGtW = [165, 268, 346, 0]
TestGtWo = [105, 217, 194, 307]

def compute_iou(bb, gt):
    # bb[0]: x1, bb[1]: y1, bb[2]: x2, bb[3]: y2
    # gt[0]: x1, gt[1]: y1, gt[2]: x2, gt[3]: y2

    # computing area of each rectangles
    bb_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    # computing the sum_area
    sum_area = bb_area + gt_area

    # find the each edge of intersect rectangle
    it = [0, 0, 0, 0]
    it[0] = max(bb[0], gt[0])
    it[1] = max(bb[1], gt[1])
    it[2] = min(bb[2], gt[2])
    it[3] = min(bb[3], gt[3])

    # judge if there is an intersect
    if it[0] >= it[2] or it[1] >= it[3]:
        return 0
    else:
        it_area = (it[2] - it[0]) * (it[3] - it[1])
        return (it_area / (sum_area - it_area))*1.0

def compute_iop(bb, gt):
    # bb[0]: x1, bb[1]: y1, bb[2]: x2, bb[3]: y2
    # gt[0]: x1, gt[1]: y1, gt[2]: x2, gt[3]: y2

    # computing area of each rectangles
    bb_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    # computing the sum_area
    sum_area = bb_area + gt_area

    # find the each edge of intersect rectangle
    it = [0, 0, 0, 0]
    it[0] = max(bb[0], gt[0])
    it[1] = max(bb[1], gt[1])
    it[2] = min(bb[2], gt[2])
    it[3] = min(bb[3], gt[3])

    # judge if there is an intersect
    if it[0] >= it[2] or it[1] >= it[3]:
        return 0
    else:
        it_area = (it[2] - it[0]) * (it[3] - it[1])
        return (it_area / bb_area)*1.0

def compute_iogt(bb, gt):
    # bb[0]: x1, bb[1]: y1, bb[2]: x2, bb[3]: y2
    # gt[0]: x1, gt[1]: y1, gt[2]: x2, gt[3]: y2

    # computing area of each rectangles
    bb_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    # computing the sum_area
    sum_area = bb_area + gt_area

    # find the each edge of intersect rectangle
    it = [0, 0, 0, 0]
    it[0] = max(bb[0], gt[0])
    it[1] = max(bb[1], gt[1])
    it[2] = min(bb[2], gt[2])
    it[3] = min(bb[3], gt[3])

    # judge if there is an intersect
    if it[0] >= it[2] or it[1] >= it[3]:
        return 0
    else:
        it_area = (it[2] - it[0]) * (it[3] - it[1])
        return (it_area / gt_area)*1.0

print('\n\n')

for cs in range(0, cs_max):
    print('Model: ' + ModelName)
    print('Confidence Score: ' + ConfScore[cs])
    print('IoU Threshold: ' + str(iou_thres))
    print('IoP Threshold: ' + str(iop_thres))
    print('IoGT Threshold: ' + str(iogt_thres))
    
    for tn in range(0, 4):
        
        print('\nTest: Test' + TestName[tn])
        
        BbPath = ModelName + '/003_Result/' + ConfScore[cs] + '/ggo_test_' + TestName[tn] + '.csv'
        GtPath = GroundTruthName + '/003_Result/' + ConfScore[cs] + '/ggo_test_' + TestName[tn]
        if tn != 3:
            BbList = []
            GtWList = []
            GtWoList = []
            with open(BbPath, newline='') as BbFile:
                BbRows = csv.reader(BbFile)
                for BbRow in BbRows:
                    BbList.append(BbRow)
            with open(GtPath + '_w.csv', newline='') as GtWFile:
                GtWRows = csv.reader(GtWFile)
                for GtWRow in GtWRows:
                    GtWList.append(GtWRow)
            with open(GtPath + '_wo.csv', newline='') as GtWoFile:
                GtWoRows = csv.reader(GtWoFile)
                for GtWoRow in GtWoRows:
                    GtWoList.append(GtWoRow)

            TpW = TpWo = Fp = FpW = FpWo = 0
            for BbLine in BbList:
                IsW = IsWo = InW = InWo = 0
                if BbLine[0][0] != 'f':
                    for GtWLine in GtWList:
                        if os.path.basename(BbLine[0]) == GtWLine[4]:
                            InW = 1
                            bb = (int(BbLine[1]), int(BbLine[2]), int(BbLine[3]), int(BbLine[4]))
                            gt = (int(GtWLine[5]), int(GtWLine[6]), int(GtWLine[7]), int(GtWLine[8]))
                            iou = compute_iou(bb, gt)
                            if iou > iou_thres:
                                GtWLine[0] = 'used'
                                TpW = TpW + 1
                                IsW = 1
                            else:
                                iop = compute_iop(bb, gt)
                                if iop > iop_thres:
                                    GtWLine[0] = 'used'
                                    TpW = TpW + 1
                                    IsW = 1
                                else:
                                    iogt = compute_iogt(bb, gt)
                                    if iogt > iogt_thres:
                                        GtWLine[0] = 'used'
                                        TpW = TpW + 1
                                        IsW = 1
                    for GtWoLine in GtWoList:
                        if os.path.basename(BbLine[0]) == GtWoLine[4]:
                            InWo = 1
                            bb = (int(BbLine[1]), int(BbLine[2]), int(BbLine[3]), int(BbLine[4]))
                            gt = (int(GtWoLine[5]), int(GtWoLine[6]), int(GtWoLine[7]), int(GtWoLine[8]))
                            iou = compute_iou(bb, gt)
                            if iou > iou_thres:
                                GtWoLine[0] = 'used'
                                TpWo = TpWo + 1
                                IsWo = 1
                            else:
                                iop = compute_iop(bb, gt)
                                if iop > iop_thres:
                                    GtWoLine[0] = 'used'
                                    TpWo = TpWo + 1
                                    IsWo = 1
                                else:
                                    iogt = compute_iogt(bb, gt)
                                    if iogt > iogt_thres:
                                        GtWoLine[0] = 'used'
                                        TpWo = TpWo + 1
                                        IsWo = 1
                    if IsW == 0 and IsWo == 0:
                        if InW == 1 and InWo == 1:
                            Fp = Fp + 1
                        elif InW == 1:
                            FpW = FpW + 1
                        elif InWo == 1:
                            FpWo = FpWo + 1
            
            print('TpW: ' + str(TpW))
            print('TpWo: ' + str(TpWo))
            print('Fp: ' + str(Fp))
            print('FpW: ' + str(Fp + FpW))
            print('FpWo: ' + str(Fp + FpWo))
            
            TestTpW[tn] = TpW
            TestTpWo[tn] = TpWo
            TestFpW[tn] = FpW
            TestFpWo[tn] = FpWo

        elif tn == 3:
            BbList = []
            GtList = []
            with open(BbPath, newline='') as BbFile:
                BbRows = csv.reader(BbFile)
                for BbRow in BbRows:
                    BbList.append(BbRow)
            with open(GtPath + '.csv', newline='') as GtFile:
                GtRows = csv.reader(GtFile)
                for GtRow in GtRows:
                    GtList.append(GtRow)

            Tp = Fp = 0
            for BbLine in BbList:
                Is = In = 0
                if BbLine[0][0] != 'f':
                    for GtLine in GtList:
                        if os.path.basename(BbLine[0]) == GtLine[4]:
                            In = 1
                            bb = (int(BbLine[1]), int(BbLine[2]), int(BbLine[3]), int(BbLine[4]))
                            gt = (int(GtLine[5]), int(GtLine[6]), int(GtLine[7]), int(GtLine[8]))
                            iou = compute_iou(bb, gt)
                            if iou > iou_thres:
                                GtLine[0] = 'used'
                                Tp = Tp + 1
                                Is = 1
                            else:
                                iop = compute_iop(bb, gt)
                                if iop > iop_thres:
                                    GtLine[0] = 'used'
                                    Tp = Tp + 1
                                    Is = 1
                                else:
                                    iogt = compute_iogt(bb, gt)
                                    if iogt > iogt_thres:
                                        GtLine[0] = 'used'
                                        Tp = Tp + 1
                                        Is = 1
                    if Is == 0:
                        if In == 1:
                            Fp = Fp + 1
            
            print('Tp: ' + str(Tp))
            print('Fp: ' + str(Fp))
            
            TestTpWo[tn] = Tp
            TestFpWo[tn] = Fp

    print('\nRecall:')
    print('Data_Ln 2D_w_patho: ')
    first = TestTpW[1] + TestTpW[2]
    second = TestGtW[1] + TestGtW[2]
    r1 = ans = Decimal(str(first/second*100)).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(first) + '/' + str(second) + ' = ' + str(ans))
    print('Data_Ln 2D_wo_patho: ')
    first = TestTpWo[1] + TestTpWo[2]
    second = TestGtWo[1] + TestGtWo[2]
    r2 = ans = Decimal(str(first/second*100)).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(first) + '/' + str(second) + ' = ' + str(ans))
    print('Data_Me 2D_wo_patho: ')
    first = TestTpWo[3]
    second = TestGtWo[3]
    r3 = ans = Decimal(str(first/second*100)).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(first) + '/' + str(second) + ' = ' + str(ans))
    print('Total: ')
    first = TestTpW[1] + TestTpW[2] + TestTpWo[1] + TestTpWo[2] + TestTpWo[3]
    second = TestGtW[1] + TestGtW[2] + TestGtWo[1] + TestGtWo[2] + TestGtWo[3]
    r4 = ans = Decimal(str(first/second*100)).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(first) + '/' + str(second) + ' = ' + str(ans))

    print('\nPrecision:')
    print('Data_Ln 2D_w_patho: ')
    first = TestTpW[1] + TestTpW[2]
    second = TestTpW[1] + TestTpW[2] + TestFpW[1] + TestFpW[2]
    p1 = ans = Decimal(str(first/second*100)).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(first) + '/' + str(second) + ' = ' + str(ans))
    print('Data_Ln 2D_wo_patho: ')
    first = TestTpWo[1] + TestTpWo[2]
    second = TestTpWo[1] + TestTpWo[2] + TestFpWo[1] + TestFpWo[2]
    p2 = ans = Decimal(str(first/second*100)).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(first) + '/' + str(second) + ' = ' + str(ans))
    print('Data_Me 2D_wo_patho: ')
    first = TestTpWo[3]
    second = TestTpWo[3] + TestFpWo[3]
    p3 = ans = Decimal(str(first/second*100)).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(first) + '/' + str(second) + ' = ' + str(ans))
    print('Total: ')
    first = TestTpW[1] + TestTpW[2] + TestTpWo[1] + TestTpWo[2] + TestTpWo[3]
    second = TestTpW[1] + TestTpW[2] + TestFpW[1] + TestFpW[2] + TestTpWo[1] + TestTpWo[2] + TestFpWo[1] + TestFpWo[2] + TestTpWo[3] + TestFpWo[3]
    p4 = ans = Decimal(str(first/second*100)).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(first) + '/' + str(second) + ' = ' + str(ans))

    print('\nF1 Score:')
    print('Data_Ln 2D_w_patho: ')

    ans = Decimal(str((2*r1*p1)/(r1+p1))).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(ans))
    print('Data_Ln 2D_wo_patho: ')
    ans = Decimal(str((2*r2*p2)/(r2+p2))).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(ans))
    print('Data_Me 2D_wo_patho: ')
    ans = Decimal(str((2*r3*p3)/(r3+p3))).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(ans))
    print('Total: ')
    ans = Decimal(str((2*r4*p4)/(r4+p4))).quantize(Decimal('0.0'), rounding = ROUND_HALF_UP)
    print(str(ans))
