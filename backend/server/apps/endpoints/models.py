from django.db import models


class Measurements(models.Model):
    '''
    The Measurements object represents measurement endpoint.

    Attributes:
        measurement_id: Unique id of the measurement
        timestamp: Time of the measurement.
        measurement_type: Acceleration (a) or Gyroscope (g)
        limp_type: Arm or Foot
        limp_side: Left or Right

        v1: first coordinate of the measured vector
        v2: second coordinate of the measured vector
        v3: third coordinate of the measured vector

        created_at: The date when endpoint was created.
    '''
    measurement_id = models.IntegerField()
    timestamp = models.BigIntegerField()
    measurement_type = models.CharField(max_length=1)
    limp_type = models.CharField(max_length=1)
    limp_side = models.CharField(max_length=1)

    max_digits = 5
    decimal_places = 3
    v1 = models.DecimalField(max_digits=max_digits, decimal_places=decimal_places)
    v2 = models.DecimalField(max_digits=max_digits, decimal_places=decimal_places)
    v3 = models.DecimalField(max_digits=max_digits, decimal_places=decimal_places)

    created_at = models.DateTimeField(auto_now_add=True, blank=True)

    # class Meta:
    #     managed = True
    #     db_table = 'endpoints_measurements'


class Endpoint(models.Model):
    '''
    The Endpoint object represents ML API endpoint.

    Attributes:
        name: The name of the endpoint, it will be used in API URL,
        owner: The string with owner name,
        created_at: The date when endpoint was created.
    '''
    name = models.CharField(max_length=128)
    owner = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)


class MLAlgorithm(models.Model):
    '''
    The MLAlgorithm represent the ML algorithm object.

    Attributes:
        name: The name of the algorithm.
        description: The short description of how the algorithm works.
        code: The code of the algorithm.
        version: The version of the algorithm similar to software versioning.
        owner: The name of the owner.
        created_at: The date when MLAlgorithm was added.
        parent_endpoint: The reference to the Endpoint.
    '''
    name = models.TextField()
    description = models.TextField()
    code = models.TextField()
    version = models.TextField()
    owner = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    parent_endpoint = models.ForeignKey(Endpoint, on_delete=models.CASCADE)

    # class Meta:
    #     managed = True
    #     db_table = 'endpoints_mlalgorithm'


class MLAlgorithmStatus(models.Model):
    '''
    The MLAlgorithmStatus represent status of the MLAlgorithm which can change during the time.

    Attributes:
        status: The status of algorithm in the endpoint. Can be: testing, staging, production, ab_testing.
        created_by: The name of creator.
        created_at: The date of status creation.
        parent_mlalgorithm: The reference to corresponding MLAlgorithm.
        parent_endpoint: The reference to corresonding Endpoint.
    '''
    status = models.CharField(max_length=128)
    active = models.BooleanField()
    created_by = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    parent_mlalgorithm = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE, related_name="status")


class MLRequest(models.Model):
    '''
    The MLRequest will keep information about all requests to ML algorithms.

    Attributes:
        input_data: The input data to ML algorithm in JSON format.
        response: The response of the ML algorithm in JSON format.
        feedback: The feedback about the response in JSON format.
        created_at: The date when request was created.
        parent_mlalgorithm: The reference to MLAlgorithm used to compute response.
    '''
    input_data = models.TextField()
    full_response = models.TextField()
    response = models.TextField()
    feedback = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    parent_mlalgorithm = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE)


class ABTest(models.Model):
    '''
    The ABTest will keep information about A/B tests.
    Attributes:
        title: The title of test.
        created_by: The name of creator.
        created_at: The date of test creation.
        ended_at: The date of test stop.
        summary: The description with test summary, created at test stop.
        parent_mlalgorithm_1: The reference to the first corresponding MLAlgorithm.
        parent_mlalgorithm_2: The reference to the second corresponding MLAlgorithm.
    '''
    title = models.TextField()
    created_by = models.CharField(max_length=128)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    ended_at = models.DateTimeField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)

    parent_mlalgorithm_1 = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE, related_name="parent_mlalgorithm_1")
    parent_mlalgorithm_2 = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE, related_name="parent_mlalgorithm_2")
