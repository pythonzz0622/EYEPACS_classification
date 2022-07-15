import tensorflow.keras.optimizers as optimizers
# import tensorflow.keras.optimizers.experimental as experimental
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt 

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


def set_classifier(hidden_layer_num  , hidden_df = None):
    if hidden_layer_num == 0:
        classifier = keras.Sequential([
            layers.Dense(2 , activation= 'softmax' , name = 'output')
            ])
    if hidden_layer_num == 1:
        classifier = keras.Sequential([
            layers.Dense(hidden_df['hidden_1']),
            layers.Dropout(hidden_df['dropout_1']),
            layers.Dense(2, activation='softmax', name='output')
        ])
    if hidden_layer_num == 2 :
        classifier = keras.Sequential([
            layers.Dense(hidden_df['hidden_1']),
            layers.Dropout(hidden_df['droput_1']),
            layers.Dense(hidden_df['hidden_2']),
            layers.Dropout(hidden_df['dropout_2']),
            layers.Dense(2, activation='softmax', name='output')
        ])
    return classifier

def save_plot(metrics , ex_name):
    fig , loss_ax = plt.subplots(figsize = (10, 6))
    du_ax = loss_ax.twinx()
    loss_ax.plot(metrics['train_loss'] ,'y' , label  = 'Train Loss')
    loss_ax.plot(metrics['val_loss'] ,  'g' , label  = 'Val_loss')
    du_ax.plot(metrics['train_acc'] , 'b' , label = 'Train_Acc')
    du_ax.plot(metrics['val_acc'] ,'r',   label = 'Val_Acc')
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss' )
    du_ax.set_ylabel('Acc' ) 

    #범례 그리기 파트
    handles , labels = du_ax.get_legend_handles_labels()
    legend1 = du_ax.legend(handles = handles , labels = labels , ncol = 2 , 
                     loc = 'upper left' , frameon=True)
    loss_ax.grid(axis="both", c="lightgray")
    fig.savefig(f'../artifact/{ex_name}_pred_val_plot.png', dpi=600 ,bbox_inches='tight')
    fig
    plt.close(fig)
    # plt.show()
