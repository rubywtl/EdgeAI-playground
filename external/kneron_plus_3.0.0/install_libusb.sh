#!/bin/sh

append_text_if_not_contain() {
    file=$1
    shift
    content=("$@")

    for ((i=0; i < ${#content[@]}; i++))
    do
        if ! grep -Fxq "${content[$i]}" ${file}
        then
            echo ${content[$i]} >> ${file}
        fi
    done
}

promote_usb_dongle_permission() {
    config_file=$1
    shift
    config_content=("$@")

    if $(test -f "${config_file}")
    then
        append_text_if_not_contain ${config_file} "${config_content[@]}"
    else
        for ((i=0; i < ${#config_content[@]}; i++))
        do
            echo ${config_content[$i]} >> ${config_file}
        done
    fi
}

# Install libusb
sudo apt install libusb-1.0-0-dev

config_file="/etc/udev/rules.d/10-local.rules"

IFS=""
eval config_content[0]='"KERNEL==\"ttyUSB*\",ATTRS{idVendor}==\"067b\",ATTRS{idProduct}==\"2303\",MODE=\"0777\",SYMLINK+=\"kneron_uart\""'
eval config_content[1]='"KERNEL==\"ttyUSB*\",ATTRS{idVendor}==\"1a86\",ATTRS{idProduct}==\"7523\",MODE=\"0777\",SYMLINK+=\"kneron_pwr\""'
eval config_content[2]='"SUBSYSTEM==\"usb\",ATTRS{product}==\"Kneron KL520\",ATTRS{idVendor}==\"3231\",ATTRS{idProduct}==\"0100\",MODE=\"0666\""'
eval config_content[3]='"SUBSYSTEM==\"usb\",ATTRS{product}==\"Kneron KL720l\",ATTRS{idVendor}==\"3231\",ATTRS{idProduct}==\"0200\",MODE=\"0666\""'
eval config_content[4]='"SUBSYSTEM==\"usb\",ATTRS{product}==\"Kneron KL720\",ATTRS{idVendor}==\"3231\",ATTRS{idProduct}==\"0720\",MODE=\"0666\""'
eval config_content[5]='"SUBSYSTEM==\"usb\",ATTRS{product}==\"Kneron KL630\",ATTRS{idVendor}==\"3231\",ATTRS{idProduct}==\"0630\",MODE=\"0666\""'
eval config_content[6]='"SUBSYSTEM==\"usb\",ATTRS{product}==\"Kneron KL730\",ATTRS{idVendor}==\"3231\",ATTRS{idProduct}==\"0732\",MODE=\"0666\""'
eval config_content[7]='"SUBSYSTEM==\"usb\",ATTRS{product}==\"Kneron KL830\",ATTRS{idVendor}==\"3231\",ATTRS{idProduct}==\"0832\",MODE=\"0666\""'


promote_usb_dongle_permission ${config_file} ${config_content[@]}

sudo udevadm control --reload-rules
sudo udevadm trigger
