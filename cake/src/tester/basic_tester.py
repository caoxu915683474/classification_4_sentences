# coding:utf-8

class BasicTester(object):
    
    def __init__(self, model_path, output_folder=None, result_type="label"):
        self.model_path = model_path
        self.model_name = "Tensorflow_Model" if not self.model_path.endswith("pkl") else "Sklearn_Model"
        self._module = __import__("cake.src.tester.%s" % self.model_name.lower(), fromlist=["%s" % self.model_name])
        dic = {"model_path":self.model_path, "output_folder":output_folder, "result_type":result_type}
        try:
            self.model = getattr(self._module, self.model_name)(**dic)
        except Exception as err:
            print("Catch error: %s" % err)
            exit(1)

    def test(self, raw_data):
        return self.model.test(raw_data)

