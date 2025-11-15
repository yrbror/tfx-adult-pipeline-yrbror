FROM prom/prometheus:v2.52.0

COPY prometheus.yml /etc/prometheus/prometheus.yml

CMD [ \
  "--config.file=/etc/prometheus/prometheus.yml", \
  "--storage.tsdb.path=/prometheus", \
  "--web.enable-lifecycle" \
]