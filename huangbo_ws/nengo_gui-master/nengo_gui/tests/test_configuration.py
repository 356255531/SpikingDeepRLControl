import time

from selenium.webdriver import ActionChains

from nengo_gui import testing_tools as tt


def test_config_persists(driver):
    tt.reset_page(driver)
    driver.execute_script("$('#Config_menu').click()")
    time.sleep(0.5)
    actions = ActionChains(driver)
    button = driver.find_element_by_id("Config_button")
    actions.move_to_element(button)
    actions.click()
    actions.perform()
    time.sleep(1.0)

    actions = ActionChains(driver)
    checkbox = driver.find_element_by_id("zoom-fonts")
    actions.move_to_element(checkbox)
    actions.click()
    actions.perform()
    time.sleep(0.5)

    val = driver.execute_script("return localStorage.getItem('ng.zoom_fonts')")
    assert val == 'true'

    actions.perform()  # do it again to go back to default (False)
    val = driver.execute_script("return localStorage.getItem('ng.zoom_fonts')")
    assert val == 'false'
