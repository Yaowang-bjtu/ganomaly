testimages = {
'C0008':{
'abnormal':
{1,48,51,100},
'normal':
{1,12,23}},

'C0007':{
'abnormal':
{1,71,335,394},
'normal':
{1,12,26,49,136}},

'C0006':{
'abnormal':
{3,114,424},
'normal':
{1,83,149}},

'C0005':{
'abnormal':
{5,65,358,394},
'normal':
{1,12,23}},

'C0004':{
'abnormal':
{103,114,358},
'normal':
{3,4,13,71}},

'C0003':{
'abnormal':
{2,278,400},
'normal':
{2,3,4}},

'C0002':{
'abnormal':
{2,388},
'normal':
{6,25,26}},

'C0001':{
'abnormal':
{1,2,3},
'normal':
{1,2}}
}


def main(channel, datatype):
    DATASET = 'C{0:04d}'.format(channel)
    for num in testimages[DATASET][datatype]:
        print(num)
    

if __name__ == '__main__':
    main(6,'abnormal')