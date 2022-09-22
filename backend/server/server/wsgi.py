import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.income_classifier.mlp import MLP  # import Multi Layer Perceptron ML algorithm

try:
    mlp = MLP()
except Exception as e:
    print("Exception while loading the algorithms, ", str(e))
# try:
#     registry = MLRegistry()  # create ML registry
#
#     ##### Multi Layer Perceptron #####
#     mlp = MLP()
#     # add to ML registry
#     registry.add_algorithm(endpoint_name="income_classifier",
#                            algorithm_object=mlp,
#                            algorithm_name="multi layer perceptron",
#                            algorithm_status="production",
#                            algorithm_version="0.0.1",
#                            owner="LevoPeti",
#                            algorithm_description="Multi Layer Perceptron with simple pre- and post-processing",
#                            algorithm_code=inspect.getsource(MLP))
#
# except Exception as e:
#     print("Exception while loading the algorithms to the registry,", str(e))
