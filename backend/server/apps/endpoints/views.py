from rest_framework import viewsets
from rest_framework import mixins

from apps.endpoints.models import Measurements
from apps.endpoints.serializers import MeasurementSerializer

from apps.endpoints.models import Endpoint
from apps.endpoints.serializers import EndpointSerializer

from apps.endpoints.models import MLAlgorithm
from apps.endpoints.serializers import MLAlgorithmSerializer

from apps.endpoints.models import MLAlgorithmStatus
from apps.endpoints.serializers import MLAlgorithmStatusSerializer

from apps.endpoints.models import MLRequest
from apps.endpoints.serializers import MLRequestSerializer

from django.db import transaction
from apps.endpoints.models import ABTest
from apps.endpoints.serializers import ABTestSerializer

import json
import pandas as pd
from numpy.random import rand
from rest_framework import views, status
from rest_framework.response import Response
from apps.ml.registry import MLRegistry
from apps.ml.income_classifier.measurement import Measurement, NotEnoughData
from server.wsgi import mlp
# from server.wsgi import registry


class MeasurementViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = MeasurementSerializer
    queryset = Measurements.objects.all()


class EndpointViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


class MLAlgorithmViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = MLAlgorithmSerializer
    queryset = MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm=instance.parent_mlalgorithm,
                                                    created_at__lt=instance.created_at,
                                                    active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])


class MLAlgorithmStatusViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
                               mixins.CreateModelMixin):
    serializer_class = MLAlgorithmStatusSerializer
    queryset = MLAlgorithmStatus.objects.all()

    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                # set active=False for other statuses
                deactivate_other_statuses(instance)

        except Exception as e:
            raise APIException(str(e))


class MLRequestViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
                       mixins.UpdateModelMixin):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()


class PredictView(views.APIView):
    def post(self, request, endpoint_name, format=None):

        algorithm_status = self.request.query_params.get("status", "production")
        algorithm_version = self.request.query_params.get("version")

        algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name, status__status=algorithm_status,
                                          status__active=True)

        if algorithm_version is not None:
            algs = algs.filter(version=algorithm_version)

        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if len(algs) != 1 and algorithm_status != "ab_testing":
            return Response(
                {"status": "Error",
                 "message": "ML algorithm selection is ambiguous. Please specify algorithm version."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 0
        if algorithm_status == "ab_testing":
            alg_index = 0 if rand() < 0.5 else 1

        algorithm_object = registry.endpoints[algs[alg_index].id]
        prediction = algorithm_object.compute_prediction(request.data)

        label = prediction["label"] if "label" in prediction else "error"
        ml_request = MLRequest(
            input_data=json.dumps(request.data),
            full_response=prediction,
            response=label,
            feedback="",
            parent_mlalgorithm=algs[alg_index],
        )
        ml_request.save()

        prediction["request_id"] = ml_request.id

        return Response(prediction)


class SaveAndPredictView(views.APIView):
    """
        {"measure": [{"limp": "f",
                     "side": "l",
                     "type": "g",
                     "values": [{"timestamp": 1224L,
                                 "vector": {"v1": 123f,
                                            "v2": 1234f,
                                            "v3": 322f
                                           }
                                },
                                {"timestamp": 1224L,
                                 "vector": {"v1": 123f,
                                            "v2": 1234f,
                                            "v3": 322f
                                           }
                                }...
                               ]
                    },
                    {"limp": "a",
                     "side": "r",
                     "type": "a",
                     "values": [{"timestamp": 1224L,
                                 "vector": {"v1": 123f,
                                            "v2": 1234f,
                                            "v3": 322f
                                           }
                                },
                                {"timestamp": 1224L,
                                 "vector": {"v1": 123f,
                                            "v2": 1234f,
                                            "v3": 322f
                                           }
                                }...
                                ]
                    }
                   ]...
        }
    """

    def post(self, request, endpoint_name, format=None):
        def load_ml_algorithm():
            return mlp
        # def old_load_ml_algorithm():
        #     algorithm_status = self.request.query_params.get("status", "production")
        #     algorithm_version = "0.0.1"  # self.request.query_params.get("version")
        #
        #     algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name, status__status=algorithm_status,
        #                                       status__active=True)
        #
        #     if algorithm_version is not None:
        #         algs = algs.filter(version=algorithm_version)
        #
        #     # TODO: we have more algs from the same type, but the need the first now
        #     # if len(algs) == 0:
        #     #     raise ValueError("status: Error, message: ML algorithm is not available")
        #     # if len(algs) != 1 and algorithm_status != "ab_testing":
        #     #     raise ValueError("status: Error, message: ML algorithm selection is ambiguous."
        #     #                      " Please specify algorithm version.")
        #
        #     # TODO
        #     # alg_index = 0
        #     # _algorithm_object = registry.endpoints[algs[alg_index].id]
        #     _algorithm_object = registry.endpoints[3]
        #     return _algorithm_object

        def write_data_into_db():
            _input_data = json.loads(request.data)
            measurement_id = _input_data["measurement_id"]
            measurement_list = list()
            for measurement_item in _input_data["measure"]:
                for value in measurement_item["values"]:
                    measurement = Measurements(
                        measurement_id=measurement_id,
                        timestamp=value["timestamp"],
                        measurement_type=measurement_item["type"],
                        limp_type=measurement_item["limp"],
                        limp_side=measurement_item["side"],
                        v1=value["vector"]["v1"],
                        v2=value["vector"]["v2"],
                        v3=value["vector"]["v3"],
                    )
                    measurement_list.append(measurement)
            Measurements.objects.bulk_create(measurement_list)
            return _input_data

        def get_meas_from_db():
            measurement_id = input_data["measurement_id"]
            left_arm_acc = Measurements.objects.filter(measurement_id=measurement_id,
                                                       measurement_type="a",
                                                       limp_type="a",
                                                       limp_side="l").values()
            left_arm_gyr = Measurements.objects.filter(measurement_id=measurement_id,
                                                       measurement_type="g",
                                                       limp_type="a",
                                                       limp_side="l").values()
            right_arm_acc = Measurements.objects.filter(measurement_id=measurement_id,
                                                        measurement_type="a",
                                                        limp_type="a",
                                                        limp_side="r").values()
            right_arm_gyr = Measurements.objects.filter(measurement_id=measurement_id,
                                                        measurement_type="g",
                                                        limp_type="a",
                                                        limp_side="r").values()
            left_foot_acc = Measurements.objects.filter(measurement_id=measurement_id,
                                                        measurement_type="a",
                                                        limp_type="f",
                                                        limp_side="l").values()
            left_foot_gyr = Measurements.objects.filter(measurement_id=measurement_id,
                                                        measurement_type="g",
                                                        limp_type="f",
                                                        limp_side="l").values()
            right_foot_acc = Measurements.objects.filter(measurement_id=measurement_id,
                                                         measurement_type="a",
                                                         limp_type="f",
                                                         limp_side="r").values()
            right_foot_gyr = Measurements.objects.filter(measurement_id=measurement_id,
                                                         measurement_type="g",
                                                         limp_type="f",
                                                         limp_side="r").values()

            def df_from_query(queries):
                df_dict = {'timestamp': list(),
                           'v1': list(),
                           'v2': list(),
                           'v3': list()}

                for query in queries:
                    for key in df_dict.keys():
                        df_dict[key].append(query[key])

                return pd.DataFrame.from_dict(df_dict)

            _meas = Measurement(measurement_id)

            _meas.measurement_dict = {
                ("left", "arm", "acc"): df_from_query(left_arm_acc),
                ("left", "arm", "gyr"): df_from_query(left_arm_gyr),
                ("left", "leg", "acc"): df_from_query(left_foot_acc),
                ("left", "leg", "gyr"): df_from_query(left_foot_gyr),
                ("right", "arm", "acc"): df_from_query(right_arm_acc),
                ("right", "arm", "gyr"): df_from_query(right_arm_gyr),
                ("right", "leg", "acc"): df_from_query(right_foot_acc),
                ("right", "leg", "gyr"): df_from_query(right_foot_gyr),
            }

            return _meas

        def get_instance():
            frequency = 25  # Hz, 40 ms
            expected_delta = (1 / frequency) * 1000  # ms
            eps = 3
            meas.check_frequency(expected_delta, eps=eps)
            meas.synchronize_measurement_dict()

            minutes = 10
            length = frequency * 60 * minutes
            keys_in_order = (("arm", "acc"),
                             ("leg", "acc"),
                             ("arm", "gyr"),
                             ("leg", "gyr"))

            _instance = list()
            for key in keys_in_order:
                diff_mean = meas.get_limb_diff_mean(key[0], key[1], length)
                ratio_mean_first = meas.get_limb_ratio_mean(key[0], key[1], length, mean_first=True)
                ratio_mean = meas.get_limb_ratio_mean(key[0], key[1], length, mean_first=False)

                _instance.append([diff_mean, ratio_mean_first, ratio_mean])

            return sum(_instance, [])

        # from time import sleep
        # sleep(3)

        try:
            input_data = write_data_into_db()
        except Exception as e:
            print(e)
            return Response({"status": "Error during write data into the db: {}".format(repr(e)), "prediction": "None"})

        try:
            meas = get_meas_from_db()
        except Exception as e:
            print(e)
            return Response({"status": "Error during get measurement: {}".format(repr(e)), "prediction": "None"})

        try:
            instance = get_instance()
        except NotEnoughData:
            return Response({"status": "No prediction, more data needed", "prediction": "None"})
        except Exception as e:
            print(e)
            return Response({"status": "Error during get instance: {}".format(repr(e)), "prediction": "None"})

        try:
            algorithm_object = load_ml_algorithm()
        except Exception as e:
            print(e)
            return Response({"status": "Error during load the algorithm: {}".format(repr(e)), "prediction": "None"})

        try:
            prediction = algorithm_object.compute_prediction([instance])
        except Exception as e:
            print(e)
            return Response({"status": "Error during make prediction: {}".format(repr(e)), "prediction": "None"})

        print(prediction)
        return Response({"status": "OK", "prediction": prediction})


class ABTestViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.CreateModelMixin, mixins.UpdateModelMixin
):
    serializer_class = ABTestSerializer
    queryset = ABTest.objects.all()

    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save()
                # update status for first algorithm

                status_1 = MLAlgorithmStatus(status="ab_testing",
                                             created_by=instance.created_by,
                                             parent_mlalgorithm=instance.parent_mlalgorithm_1,
                                             active=True)
                status_1.save()
                deactivate_other_statuses(status_1)
                # update status for second algorithm
                status_2 = MLAlgorithmStatus(status="ab_testing",
                                             created_by=instance.created_by,
                                             parent_mlalgorithm=instance.parent_mlalgorithm_2,
                                             active=True)
                status_2.save()
                deactivate_other_statuses(status_2)

        except Exception as e:
            raise APIException(str(e))


from django.db.models import F
import datetime


class StopABTestView(views.APIView):
    def post(self, request, ab_test_id, format=None):

        try:
            ab_test = ABTest.objects.get(pk=ab_test_id)

            if ab_test.ended_at is not None:
                return Response({"message": "AB Test already finished."})

            date_now = datetime.datetime.now()
            # alg #1 accuracy
            all_responses_1 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_1,
                                                       created_at__gt=ab_test.created_at,
                                                       created_at__lt=date_now).count()
            correct_responses_1 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_1,
                                                           created_at__gt=ab_test.created_at, created_at__lt=date_now,
                                                           response=F('feedback')).count()
            accuracy_1 = correct_responses_1 / float(all_responses_1)
            print(all_responses_1, correct_responses_1, accuracy_1)

            # alg #2 accuracy
            all_responses_2 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_2,
                                                       created_at__gt=ab_test.created_at,
                                                       created_at__lt=date_now).count()
            correct_responses_2 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_2,
                                                           created_at__gt=ab_test.created_at, created_at__lt=date_now,
                                                           response=F('feedback')).count()
            accuracy_2 = correct_responses_2 / float(all_responses_2)
            print(all_responses_2, correct_responses_2, accuracy_2)

            # select algorithm with higher accuracy
            alg_id_1, alg_id_2 = ab_test.parent_mlalgorithm_1, ab_test.parent_mlalgorithm_2
            # swap
            if accuracy_1 < accuracy_2:
                alg_id_1, alg_id_2 = alg_id_2, alg_id_1

            status_1 = MLAlgorithmStatus(status="production",
                                         created_by=ab_test.created_by,
                                         parent_mlalgorithm=alg_id_1,
                                         active=True)
            status_1.save()
            deactivate_other_statuses(status_1)
            # update status for second algorithm
            status_2 = MLAlgorithmStatus(status="testing",
                                         created_by=ab_test.created_by,
                                         parent_mlalgorithm=alg_id_2,
                                         active=True)
            status_2.save()
            deactivate_other_statuses(status_2)

            summary = "Algorithm #1 accuracy: {}, Algorithm #2 accuracy: {}".format(accuracy_1, accuracy_2)
            ab_test.ended_at = date_now
            ab_test.summary = summary
            ab_test.save()

        except Exception as e:
            return Response({"status": "Error", "message": str(e)},
                            status=status.HTTP_400_BAD_REQUEST
                            )
        return Response({"message": "AB Test finished.", "summary": summary})
