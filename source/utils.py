import tensorflow.keras.optimizers as optimizers
# import tensorflow.keras.optimizers.experimental as experimental
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt 
import cv2 
import os 

SGD = optimizers.SGD(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=False,
    name='SGD'
)

RMSProp = optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name='RMSprop'
)

Adam = optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam'
)

NAdam = optimizers.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name='Nadam'
)

Adamax = optimizers.Adamax(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name='Adamax'
)
# AdamW = experimental.AdamW(
#     learning_rate=0.001,
#     weight_decay=0.004,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-07,
#     amsgrad=False,
#     clipnorm=None,
#     clipvalue=None,
#     global_clipnorm=None,
#     use_ema=False,
#     ema_momentum=0.99,
#     ema_overwrite_frequency=None,
#     jit_compile=True,
#     name='AdamW'
# )

optimizer_dict = {'SGD' : SGD ,'RMSProp' : RMSProp  , 'Adam' : Adam , 'NAdam' : NAdam , 'Adamax' : Adamax }
def set_optimizer(optimizer_name , **kwargs):
    optimizer =  optimizer_dict[optimizer_name]
    for k , v in kwargs.items():
        setattr(optimizer , k  , v )
    return optimizer

# train , val acc , loss를 그리는 함수
def save_plot(metrics , ex_name , dir ):
    fig , loss_ax = plt.subplots(figsize = (10, 6))
    du_ax = loss_ax.twinx()
    loss_ax.plot(metrics['train_loss'] ,'y' , label  = 'Train Loss')
    loss_ax.plot(metrics['val_loss'] ,  'g' , label  = 'Val_loss')
    du_ax.plot(metrics['train_acc'] , 'b' , label = 'Train_Acc')
    du_ax.plot(metrics['val_acc'] ,'r',   label = 'Val_Acc')
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss' )
    du_ax.set_ylabel('Acc' ) 

    #make_legend
    handles , labels = du_ax.get_legend_handles_labels()
    legend1 = du_ax.legend(handles = handles , labels = labels , ncol = 2 , 
                     loc = 'upper left' , frameon=True)

    handles , labels = loss_ax.get_legend_handles_labels()
    legend1 = loss_ax.legend(handles = handles , labels = labels , ncol = 2 , 
                     loc = 'lower left' , frameon=True)

    loss_ax.grid(axis="both", c="lightgray")
    os.makedirs(dir , exist_ok= True)
    fig.savefig(f'{dir}{ex_name}_pred_val_plot.png', dpi=600 ,bbox_inches='tight')
    plt.close(fig)


def imgread(image_path , to_gray = False):
    if to_gray == True:
        image = cv2.imread(image_path ,  cv2.IMREAD_GRAYSCALE)
    elif to_gray == False:
        image = cv2.imread(image_path , cv2.IMREAD_COLOR )
        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    return image 

def find_boundary(x):
    for idx , i in enumerate(x):
        if i != 255:
            return idx