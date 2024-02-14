from confluent_kafka import Producer
import json
import time


def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))


def produce_job_information(producer, topic, job_data):
    """ Produce job information to Kafka topic. """
    job_json = json.dumps(job_data)
    producer.produce(topic, value=job_json, callback=delivery_report)


def main():
    kafka_bootstrap_servers = 'localhost'
    kafka_topic = 'host-1'

    # Set up Kafka producer configuration
    producer_conf = {
        'bootstrap.servers': kafka_bootstrap_servers,
        'client.id': 'job-producer'
    }

    # Create Kafka producer
    producer = Producer(producer_conf)

    # Example job data (replace with your actual job information)
    job_data1 = {"resources": {"cpus": 1, "gpus": 0, "nics": 0}, "commands": ["python3 -m agents.jobs.simple_job"]}
    job_data2 = {"resources": {"cpus": 2, "gpus": 0, "nics": 1}, "commands": ["python3 -m agents.jobs.simple_job"]}

    try:
        # Produce job information to Kafka topic
        for x in range(10):
            produce_job_information(producer, kafka_topic, job_data1)
            produce_job_information(producer, kafka_topic, job_data2)

        # Wait for any outstanding messages to be delivered and delivery reports received
        producer.flush()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
