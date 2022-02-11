from flask import render_template, flash, session,redirect,url_for,request
from os.path import join, dirname, realpath
import os
from app import app,db
from .forms import UserForm,goodsForm,checkForm,checkForm2
from . import models
import datetime
from datetime import timedelta
import json
import random
from flask import make_response,jsonify
import socket
import uuid

@app.route('/')
def statu():
    return render_template('hello_world.html', username="Shan")
'''
@celery.task(bind=True)
def long_task(self):
    """Background task that runs a long function with progress reports."""
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    total = random.randint(10, 50)
    for i in range(total):
        if not message or random.random() < 0.25:
            message = '{0} {1} {2}...'.format(random.choice(verb),
                                              random.choice(adjective),
                                              random.choice(noun))
        self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': total,
                                'status': message})
        time.sleep(1)
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': 42}

@app.route('/longtask', methods=['POST'])
def longtask():
    task = long_task.apply_async()
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}
@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)





db.create_all()
    
@app.route('/', methods=['GET', 'POST'])
def regist():
    app.logger.info("user register")
    form = UserForm()
    
    if form.validate_on_submit():
        for p in models.User.query.all():
            if p.name==form.name.data :
                app.logger.error("user use a username existed")
                return render_template('register.html',title='Regist',form=form,error1='This name has been used')
            elif p.email==form.email.data:
                app.logger.error("user use an email existed")
                return render_template('register.html',title='Regist',form=form,error2='This email has been used')
        if form.trader.data=='T':
            app.logger.info("user choose trader")
            p =models.User(name=form.name.data,gender=form.gender.data, email=form.email.data, password=form.password.data ,trader=True )
        else:
            app.logger.info("user choose normal user")
            p =models.User(name=form.name.data,gender=form.gender.data, email=form.email.data, password=form.password.data ,trader=False )
        if form.remember.data=='T':
            app.logger.warning("user want to use cookie")
            app.config['PERMANENT_SESSION_LIFETIME']=timedelta(days=7)
        else:
            app.logger.warning("user don't want to use cookie")
        session['administrator']=False
        session['user']=form.name.data#试试todolist
        session['trader']=p.trader
        session['id']=p.id
        db.session.add(p)
        db.session.commit()
        flash('Welcome %s'%(form.gender.data))
        app.logger.info("User successfully registered")
        return redirect(url_for('main'))
    return render_template('register.html',title='Regist',form=form, trader=session.get('trader'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    app.logger.info("user logs in")
    form = checkForm()
    if form.validate_on_submit():
        for p in models.User.query.all():
            if p.name==form.name.data:
                if p.email!=form.email.data:
                    app.logger.error("user name can't match email")
                    return render_template('login.html',form=form,error2='No such email', trader=session.get('trader'),administrator=session.get('administrator') )
                elif p.password!=form.password.data:
                    app.logger.error("user name can't match password")
                    return render_template('login.html',form=form,error3='wrong password', trader=session.get('trader'),administrator=session.get('administrator') )
                else:
                    if form.remember.data=='T':
                        app.logger.warning("user want to use cookie")
                        app.config['PERMANENT_SESSION_LIFETIME']=timedelta(days=7)
                    else:
                        app.logger.warning("user don't want to use cookie")
                    if form.name.data==models.User.query.all()[0].name:
                        session['administrator']=True
                    else:
                        session['administrator']=False
                    session['user']=form.name.data
                    session['trader']=p.trader
                    session['id']=p.id
                    app.logger.info("User successfully log in")
                    return redirect(url_for('main'))
        app.logger.error("no such user name")
        return render_template('login.html',form=form,error1='No such user name', trader=session.get('trader'),administrator=session.get('administrator') )
    return render_template('login.html',form=form, trader=session.get('trader'),administrator=session.get('administrator'))
    

@app.route('/show_all')
def show_all():
    app.logger.warning("user views important information")
    if not session.get('administrator'):
        return redirect(url_for('regist'))
    return render_template('show_all.html', Users = models.User.query.all() ,administrator=session.get('administrator') )
'''
@app.route('/show_all2')
def show_all2():
   return render_template('main.html', Users = models.todolists.query.all() )
   '''
@app.route('/main')
def main():
    app.logger.info("main page")
    if len(models.User.query.filter_by(trader=True).all())!=0:
        num=len(models.goods.query.filter(models.goods.User_id!=session.get('id'),models.goods.saled==False).group_by('name','User_id','description','url').all())
        rang=num
        if num>12:
            rang=12
        rs=random.sample(range(0,num),rang)
        randoms=[models.goods.query.filter(models.goods.User_id!=session.get('id'),models.goods.saled==False).group_by('name','User_id','description','url')[x] for x in rs]

        i=0
        total=[]
        types=['Electronics','Book','Game','Handmade','Sport']
        while i<5:
            num1=len(models.goods.query.filter(models.goods.User_id!=session.get('id'),models.goods.classify==types[i],models.goods.saled==False).group_by('name','User_id','description','url').all())
            rang1=num1
            if num1>4:
                rang1=4
            rs1=random.sample(range(0,num1),rang1)
            randoms0=[models.goods.query.filter(models.goods.User_id!=session.get('id'),models.goods.saled==False,models.goods.classify==types[i]).group_by('name','User_id','description','url')[x] for x in rs1]
            if(len(randoms0)>0):
                total.append(randoms0)
            i+=1
        return render_template('main.html', Users =models.User.query.filter(models.User.trader==True,models.User.name!=session.get('user')).all()
                               ,id=session.get('user'),trader=session.get('trader'),randoms=randoms,i=0,total=total,administrator=session.get('administrator') )
    
    return render_template('main.html', message = 'Your todo-list is empty now, please create a task',id=session.get('user'),trader=session.get('trader'),administrator=session.get('administrator')  )

@app.route('/create',methods=['GET', 'POST'] )
def create():
    app.logger.info("create new commidity")
    if not (models.User.query.all() and session.get('user')):
        return redirect(url_for('regist'))
    p=models.User.query.filter_by(name=session.get('user'))[0]
    if p.trader==False:
        return redirect(url_for('main'))
    form = goodsForm()
    if form.validate_on_submit():
            num=form.number.data
            filename = form.file.data.filename
            paths=join(dirname(realpath(__file__)),"static/photo" )
            path_list=os.listdir(paths)
            ii=0
            newname=filename
            while newname in path_list:
                newname=str(ii)+'_'+filename
                ii+=1

            filename='static/photo/'+newname
            urls='photo/'+newname
            form.file.data.save(join(dirname(realpath(__file__)),filename ))
            
            for i in range(num):
                p1=models.goods(name=form.name.data,User_id=p.id,description=form.description.data,url=urls,price=form.price.data,classify=form.classify.data)
                
                db.session.add(p1)
                
                db.session.commit()
                p2=models.log(time=datetime.datetime.now(),username=p.name,action='Trader '+p.name+' add a good named '+form.name.data,goodname=p1.name,host=socket.gethostname(),ip=socket.gethostbyname(socket.gethostname()),mac=get_mac_address())
                db.session.add(p2)
                db.session.commit()
            app.logger.warning("user adds commidity sucessfully")
            return redirect(url_for('goodcontrol'))
    return render_template('create.html',form=form, trader=session.get('trader'),administrator=session.get('administrator') )






@app.route('/respond', methods=['POST'])
def respond():
    app.logger.warning("user seraches commodities")
    data = json.loads(request.data.decode("utf-8"))
    response = data.get('response')
    tasklist='<table  cellspacing="10" style="margin:auto">'+'<thead><tr><th>Name</th><th>pic</th><th>description</th><th>price</th></tr></thead>'
    for p in models.goods.query.filter(models.goods.User_id!=session.get('id'),models.goods.saled==False).group_by('name','User_id','description','url','price'):
        if response in p.name:
            tasklist=(tasklist+'<tr><th>'+p.name+"</th><th><img src="+url_for('static', filename=p.url )+" width='80px' height='80px' /></th><th>"+p.description+'</th><th>'+str(p.price)+'</th><th><a href=/buygood/'
                      +str(p.id)+"><button >buy goods</button></a></th></tr>")
    tasklist=tasklist+'</table>'
    return json.dumps({'status': 'OK', 'response': tasklist})

@app.route('/respond2', methods=['POST'])
def respond2():
    data = json.loads(request.data.decode("utf-8"))
    response = data.get('response')
    changeid=int(response)
    for p in models.todolists.query.all():
        if changeid==p.id:
            if p.finish==None:
                p.finish=True
            elif p.finish==False:
                p.finish=True
            elif p.finish==True:
                p.finish=False
        db.session.commit()
    return json.dumps({'status': 'OK', 'response': changeid})

@app.route('/respond3', methods=['POST'])
def respond3():
    data = json.loads(request.data.decode("utf-8"))
    response = data.get('response')
    changeid=int(response)
    aaa=''
    for p in models.todolists.query.all():
        if changeid==p.id:
            aaa=aaa+p.task+' '+str(p.deadline)+' '+p.description
            return json.dumps({'status': 'OK', 'response': aaa})

@app.route('/search', methods=['GET', 'POST'])
def search():
    app.logger.info("user in search page")
    if not (len(models.User.query.all())!=0 and session.get('user')):
        return redirect(url_for('regist'))
    return render_template('search.html',trader=session.get('trader'))


@app.route('/finished')
def finished():
    if not (models.User.query.all() and session.get('user')):
        return redirect(url_for('regist'))
    if models.User.query.all():
        for p in models.User.query.all():
            if p.name==session.get('user'):
                if p.todolist:
                    return render_template('finished.html', Users = models.todolists.query.filter_by(finish=True,User_id=p.id) )
    return render_template('finished.html', message = 'Your todo-list is empty now, please create a task' )


@app.route('/unfinished')
def unfinished():
    if not (models.User.query.all() and session.get('user')):
        return redirect(url_for('regist'))
    if models.User.query.all():
        for p in models.User.query.all():
            if p.name==session.get('user'):
                if p.todolist:
                    return render_template('unfinished.html', Users = models.todolists.query.filter_by(finish=False,User_id=p.id) )
    return render_template('unfinished.html', message = 'Your todo-list is empty now, please create a task' )


@app.route('/delete')
def delete():
    app.logger.warning("user deletes database")
    if not session.get('administrator'):
        return redirect(url_for('regist'))
    for p in models.goods.query.all():
        db.session.delete(p)
    for p in models.record.query.all():
        db.session.delete(p)
    db.session.commit()
    return redirect(url_for('goodcontrol'))

@app.route('/good/<idd>')#mode buygood for user and trader
def good(idd):
    app.logger.info("user views commodity")
    if not (models.User.query.all() and session.get('user')):
        return redirect(url_for('regist'))
    if len(models.User.query.filter_by(id=idd).all())==0 or models.User.query.filter_by(id=idd)[0].trader==False:
        return redirect(url_for('main'))
    return render_template('goodscontrol.html', Users =models.User.query.filter_by(id=idd)[0].good.group_by('name','User_id','description','url'),mode='buygood',administrator=session.get('administrator') )

@app.route('/goodcontrol',methods=['GET', 'POST'] )#mode goodcontrol for trader
def goodcontrol():
    app.logger.info("user controls commodity")
    if not (models.User.query.all() and session.get('user')):
        return redirect(url_for('regist'))
    if session.get('trader')==True:
        return render_template('goodscontrol.html', Users =models.User.query.filter_by(name=session.get('user'))[0].good.group_by('name','User_id','description','url'),mode='goodcontrol',administrator=session.get('administrator') )
    else:
        return redirect(url_for('main'))
'''
@app.route('/buygood/<ids>')
def buygood(ids):
    aa=models.goods.query.filter_by(id=ids)[0]
    aa.saled=True
    a=aa.User_id
    b=models.User.query.filter_by(name=session.get('user'))[0].id
    models.goods.query.filter_by(id=ids)[0].User_id=b
    p1=models.record(goodid=ids,traderid=a,userid=b)
    db.session.add(p1)
    db.session.commit()
    return redirect(url_for('good',idd=a))
'''
@app.route('/goodcontrol/<ids>',methods=['GET', 'POST'] )#mode goodcontrol for trader
def goodcontrols(ids):
    app.logger.info("user controls commodity")
    if not (models.User.query.all() and session.get('user')):
        return redirect(url_for('regist'))
    if session.get('trader')==False:
        return redirect(url_for('main'))
    good=models.goods.query.filter_by(id=ids)[0]
    goodlist=models.goods.query.filter_by(User_id=good.User_id,name=good.name,url=good.url)
    remain=len(goodlist.all())
    return render_template('goodsale.html', good=good,remain=str(remain),administrator=session.get('administrator') )
    

@app.route('/buygood/<ids>',methods=['POST','GET'])
def buygood1(ids):
    app.logger.warning("user buy commodity")
    if not (models.User.query.all() and session.get('user')):
        return redirect(url_for('regist'))
    if session.get('trader')==True:
        return redirect(url_for('main'))
    form=checkForm2()
    good=models.goods.query.filter_by(id=ids)[0]
    goodlist=models.goods.query.filter_by(User_id=good.User_id,name=good.name,url=good.url)
    remain=len(goodlist.all())
    b=models.User.query.filter_by(name=session.get('user'))[0].id
    if form.validate_on_submit():
        if form.number.data > remain:
            return render_template('goodsale.html', good=good,remain=str(remain),error='Purchase quantity exceeds inventory ',form=form)
        for i in range(form.number.data):
            idd=goodlist[0].id
            tid=goodlist[0].User_id
            p2=models.log(time=datetime.datetime.now(),username=session.get('user'),action='User '+session.get('user')+' buy a good named '+goodlist[0].name+' from trader with id '+str(tid),goodname=goodlist[0].name,host=socket.gethostname(),ip=socket.gethostbyname(socket.gethostname()),mac=get_mac_address())
            goodlist[0].saled=True
            goodlist[0].User_id=b
            
            p1=models.record(goodid=idd,traderid=tid,userid=b)
            db.session.add(p2)
            db.session.add(p1)
            db.session.commit()
        return redirect(url_for('goodlist'))
    return render_template('goodsale.html', good=good,remain=str(remain),form=form ,administrator=session.get('administrator') )

@app.route('/record')
def record():
    if not session.get('administrator'):
        return redirect(url_for('regist'))
    app.logger.warning("user views record")
    return render_template('record.html', Users =models.record.query.all(),administrator=session.get('administrator')  )
@app.route('/log')
def log():
    if not session.get('administrator'):
        return redirect(url_for('regist'))
    app.logger.warning("user views logging")
    return render_template('log.html', Users =models.log.query.filter_by().order_by(models.log.username,models.log.time),administrator=session.get('administrator')  )

@app.route('/goodlist',methods=['GET', 'POST'] )
def goodlist():
    app.logger.info("user views commodity")
    if not (models.User.query.all() and session.get('user')):
        return redirect(url_for('regist'))
    if session.get('trader')==True:
        return redirect(url_for('main'))
    return render_template('goodscontrol.html', Users =models.User.query.filter_by(name=session.get('user'))[0].good,mode='refunds',administrator=session.get('administrator')  )

@app.route('/refunds/<idss>',methods=['GET', 'POST'] )
def refunds(idss):
    app.logger.warning("user refunds commodity")
    if not (models.User.query.all() and session.get('user')):
        return redirect(url_for('regist'))
    rec=models.record.query.filter_by(goodid=idss)[0]
    p2=models.log(time=datetime.datetime.now(),username=session.get('user'),action='User '+session.get('user')+' refund a good named '+models.goods.query.filter_by(id=idss)[0].name+' to trader with id '+str(rec.traderid),goodname=models.goods.query.filter_by(id=idss)[0].name,host=socket.gethostname(),ip=socket.gethostbyname(socket.gethostname()),mac=get_mac_address())
    models.goods.query.filter_by(id=idss)[0].saled=False
    models.goods.query.filter_by(id=idss)[0].User_id=rec.traderid
    db.session.delete(models.record.query.filter_by(goodid=idss)[0])
    db.session.add(p2)
    db.session.commit()
    return redirect(url_for('goodlist'))

def get_mac_address():
        mac=uuid.UUID(int = uuid.getnode()).hex[-12:]
        return ":".join([mac[e:e+2] for e in range(0,11,2)])
'''