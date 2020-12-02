variable "zone" {
  type = string
}
variable "project" {
  type = string
}

variable BUCKET {
  type = string
}

provider "google" {
  project = var.project
  region  = join("-", slice(split("-", var.zone), 0, 2)) // Just turns something like foo-west1-4 -> foo-west1
}

resource "google_compute_instance_template" "default" {
  name_prefix = "tpupod-"
  description = "This template is used to create pytorch tpu pod workers"

  // machine_type = "n2-standard-64" // for quota need to be < 300
  machine_type = "n2-custom-72-524288"

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
  }

  // Create a new boot disk from an image
  disk {
    source_image = "projects/ml-images/global/images/debian-9-torch-xla-v20201029"
    auto_delete  = true
    boot         = true
    disk_size_gb = "100"
  }


  network_interface {
    subnetwork = "default"
    access_config {
      nat_ip = ""
    }
  }
  metadata_startup_script = templatefile("workerboot.sh.tmpl", { BUCKET = var.BUCKET, TPU = "mytpu" })
  service_account {
    scopes = ["cloud-platform"] # cloud platform is not best practice. Need storage read, compute read/ssh
  }
  lifecycle {
    create_before_destroy = true
  }

  metadata = {
    oslogin = "TRUE"
  }
}

resource "google_compute_instance_group_manager" "instance_group_manager" {
  name               = "tpupod-instance-group"
  base_instance_name = "tpupod"
  zone               = var.zone
  target_size        = 4 # 32/8
  version {
    instance_template = google_compute_instance_template.default.id
  }
  update_policy {
    type              = "PROACTIVE"
    minimal_action    = "REPLACE"
    max_surge_percent = 100
    min_ready_sec     = 60
  }
}
resource "random_id" "instance_id" {
  byte_length = 2
}
resource "google_compute_instance" "profiler" {
  name         = "torch-profile-vm-${random_id.instance_id.hex}"
  machine_type = "e2-standard-16"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ml-images/debian-10-tf-2-3-v20200803"
    }
  }

  // Make sure flask is installed on all new instances for later steps
  metadata_startup_script = "pip3 install --upgrade 'cloud-tpu-profiler>=2.3.0';pip3 install --upgrade -U 'tensorboard>=2.3'; pip3 install --upgrade -U 'tensorflow>=2.3'"
  service_account {
    scopes = ["cloud-platform"] # cloud platform is not best practice. Need storage read, compute read/ssh
  }
  allow_stopping_for_update = true
  network_interface {
    network = "default"

    access_config {
      nat_ip = ""
    }
  }
}


resource "google_compute_firewall" "default" {
  name    = "allow-ssh-ingress-from-iap"
  network = "default"

  allow {
    protocol = "icmp"
  }

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["35.235.240.0/20", "0.0.0.0"] // https://cloud.google.com/iap/docs/using-tcp-forwarding#gcloud
}

resource "google_compute_firewall" "tensorboard" {
  name    = "allow-tensorboard"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["6006"]
  }
  source_ranges = ["0.0.0.0/0"] // TOOD: lock this down to smaller group 
}

resource "google_tpu_node" "tpu" {
  name               = "mytpu"
  zone               = var.zone
  accelerator_type   = "v2-32"
  tensorflow_version = "pytorch-1.7"
}

resource "google_monitoring_dashboard" "pytorch" {
  dashboard_json = file("dashboard.json")
}



