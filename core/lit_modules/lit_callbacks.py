
import  pytorch_lightning.callbacks as  pl_callbacks




def callback_model_checkpoint(dirpath, filename, monitor, mode, save_top_k=1, save_last=True, verbose=True, \
                                                                 every_n_epochs=1, every_n_train_steps=0, **kwargs):
    """
    Callback for model checkpointing. 

    Parameters
    ----------
    `dirpath` - str: 
        The directory where the checkpoints will be saved.
    `filename` - str: 
        The filename of the checkpoint.
    `mode` - str: 
        The mode of the monitored metric. Can be either 'min' or 'max'.
    `monitor` - str: 
        The metric to be monitored.
    `save_top_k` - int: 
        The number of top checkpoints to save.
    `save_last` - bool: 
        If True, the last checkpoint will be saved.
    `verbose` - bool: 
        If True, the callback will print information about the checkpoints.
    `every_n_epochs` - int: 
        The period of the checkpoints.
    """
    return pl_callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
        every_n_epochs=every_n_epochs,
        every_n_train_steps=every_n_train_steps,
        verbose=verbose,
        **kwargs
    )