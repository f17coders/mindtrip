package com.a303.consultms.domain.channel.repository;

import com.a303.consultms.domain.channel.Channel;
import java.util.List;
import lombok.AllArgsConstructor;
import org.checkerframework.checker.units.qual.C;
import org.springframework.data.domain.Sort;
import org.springframework.data.domain.Sort.Direction;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Repository;

@Repository
@AllArgsConstructor
public class ChannelRepositoryImpl implements ChannelCustomRepository{

    private final MongoTemplate mongoTemplate;
    @Override
    public List<Channel> findListBySenderOrReceiver(String memberId) {
        Criteria criteria = new Criteria().orOperator(
            Criteria.where("sender.memberId").is(memberId),
            Criteria.where("receiver.memberId").is(memberId)
        );
        Query query = new Query(criteria)
            .with(Sort.by(Direction.DESC, "createTime"));

        return mongoTemplate.find(query, Channel.class);
    }

    @Override
    public List<Channel> findBySender(String senderId){
        Criteria criteria = new Criteria().orOperator(
            new Criteria().andOperator(
                Criteria.where("sender.memberId").is(senderId)
            )
        );
        Query query = new Query(criteria);
        return mongoTemplate.find(query, Channel.class);
    }

    @Override
    public Channel findBySenderOrReceiver(String senderId, String receiverId) {
        Criteria criteria = new Criteria().orOperator(
            new Criteria().andOperator(
                Criteria.where("sender.memberId").is(senderId),
                Criteria.where("receiver.memberId").is(receiverId)
            ),
            new Criteria().andOperator(
                Criteria.where("sender.memberId").is(receiverId),
                Criteria.where("receiver.memberId").is(senderId)
            )
        );
        Query query = new Query(criteria);
        return mongoTemplate.findOne(query, Channel.class);
    }
}
