import PySimpleGUI as sg

#sg.theme('DarkAmber')   # デザインテーマの設定

# ウィンドウに配置するコンポーネント
layout = [  [sg.Text('This is line1')],
            [sg.Text('This is line2'), sg.InputText()],
            [sg.Button('OK'), sg.Button('Cancel')] ]

# ウィンドウの生成
window = sg.Window('Sample', layout)

# イベントループ
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    elif event == 'OK':
        print('Value： ', values[0])

window.close()
