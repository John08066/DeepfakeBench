class Registry(object):
    def __init__(self):
        self.data = {}
    
    def register_module(self, module_name=None):
        def _register(cls): #单下划线 约定含义：内部私有函数，不希望外部直接调用，  cls 是类本身，还没有实例化，不是实例对象
            name = module_name
            if module_name is None:
                name = cls.__name__
            self.data[name] = cls #Registry实例保存数据
            return cls
        return _register # 类被import创建看见装饰器模块最终返回 XceptionDetector = _register(XceptionDetector)
    
    def __getitem__(self, key):
        return self.data[key] #实现魔术方法，使得可以类似字典使用：DETECTOR['xception'] = DETECTOR.data['xception']


# 下面保证了在整个项目中，DETECTOR、BACKBONE、TRAINER、LOSSFUNC 都是单例模式的 Registry 实例，方便在不同模块之间共享注册的类。   
BACKBONE = Registry()
DETECTOR = Registry()   # DETECTOR 本质上就是一个 Registry 实例 ，import执行后，这时候只是把Xception类本身放进去，还没有实例化 Xception。
TRAINER  = Registry()
LOSSFUNC = Registry()

# 完整时间线：registry.py被执行 → DETECTOR=Registry() → Registry.__init__() →  DETECTOR.data={} → 之后xception_detector.py 被import会执行装饰器
# → @DETECTOR.register_module('xception') → Registry.register_module(cls) → _register(cls) → DETECTOR.data['xception'] = XceptionDetector