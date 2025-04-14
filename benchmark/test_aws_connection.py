import boto3


def test_aws_connection():
    try:
        client = boto3.client("sts")
        response = client.get_caller_identity()
        print("Successfully connected to AWS!")
        print(f"Account: {response['Account']}, ARN: {response['Arn']}")
    except Exception as e:
        print(f"Failed to connect to AWS: {e}")


if __name__ == "__main__":
    test_aws_connection()
